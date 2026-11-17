"""Four-rank CPU/Gloo controller integration with real tiny Qwen3/PEFT updates.

CUDA setup/metadata are redirected to CPU and vLLM is replaced with fixed
synthetic response fixtures whose log probabilities come from a real copy of
the current actor. This exercises orchestration/checkpoints, not GPU serving,
on-policy sampling performance, convergence or pretrained-model accuracy.
"""
import importlib.util
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_actor_cpu_integration import AsciiTokenizer


def loop_worker(rank, init_path, model_path, data_path, out, arm, mode, epochs, resume, interrupt,
                stable_buckets=False, batch_size=4, keep_all_checkpoints=False,
                resume_state_retention='all', allow_epoch_extension=False,
                warmup_steps=0, kl_coefficient=0., learner_microbatch_size=1):
    import trainer
    import trainer_distributed
    import trainer_resumable
    import experiment_credit
    import resumable_checkpoint as checkpoints
    from transformers import AutoTokenizer
    from sampler_worker import SamplerClient
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+init_path, rank=rank, world_size=4)
    tok = AsciiTokenizer()
    load_actor = trainer.load_actor
    reference_features = trainer.features
    feature_calls = []
    def tracked_reference_features(*args, **kwargs):
        feature_calls.append(True)
        return reference_features(*args, **kwargs)
    comparison_guard = (patch.object(experiment_credit, 'compute_group_credit',
                        side_effect=AssertionError('GRPO must not compute comparison credits'))
                        if arm == 'grpo' else nullcontext(None))
    capture_rng, restore_rng = checkpoints.capture_rng, checkpoints.restore_rng
    module = trainer_distributed if mode == 'smoke' else trainer_resumable
    ddp_models = []

    def cpu_actor(model, adapter=None, **kwargs):
        kwargs['device'] = 'cpu'
        return load_actor(model, adapter, **kwargs)

    def cpu_ddp(model, **kwargs):
        # Test-only diagnostic: unused-parameter detection disables rebuilding
        # buckets, so fresh and resumed reducers retain the same tensor layout.
        wrapped = DistributedDataParallel(model, broadcast_buffers=False,
                                          find_unused_parameters=stable_buckets,
                                          static_graph=False)
        ddp_models.append(wrapped)
        return wrapped

    class FixedResponseSampler:
        def generate(self, request, **kwargs):
            if interrupt and request['request_id'] == 'update_1':
                raise RuntimeError('Intentional test interruption after checkpoint one')
            actor = load_actor(model_path, request['adapter_path'], device='cpu', finetune_mode='lora')
            actor.eval()
            texts = ['Step 1: Add 1 and 1.\nStep 2: Get 2.\nFinal Answer: \\boxed{2}',
                     'Step 1: Double one.\nStep 2: Obtain 2.\nFinal Answer: \\boxed{2}',
                     'Step 1: Guess a number.\nStep 2: Say 3.\nFinal Answer: \\boxed{3}']
            rows = []
            for prompt in request['prompts']:
                samples = []
                for j in range(request['n']):
                    text = texts[j % len(texts)]
                    ids = tok.encode(text)
                    with torch.no_grad(): logps = trainer.selected_logps(actor, prompt['prompt_token_ids'], ids).tolist()
                    samples.append(dict(sample_index=j, token_ids=ids, logprobs=logps, text=text,
                                        finish_reason='stop', stop_reason=None))
                rows.append(dict(id=prompt['id'], prompt_token_ids=prompt['prompt_token_ids'],
                                 max_tokens=min(request['max_tokens'],8192-len(prompt['prompt_token_ids'])), samples=samples))
            del actor
            return dict(rows=rows, generation_seconds=0.)
        def close(self):
            pass

    args = ['driver', mode, '--model', 'qwen4b', '--model-path', model_path,
            '--finetune-mode', 'lora', '--wandb-mode', 'disabled', '--validation-mode', 'off',
            '--data', data_path, '--data-sha256', trainer.digest(data_path), '--out', out,
            '--arm', arm, '--batch-size', str(batch_size), '--epochs', str(epochs), '--rollouts', '3',
            '--learning-rate', '.01', '--seed', '42', '--smoke-cap', '128']
    args += ['--warmup-steps', str(warmup_steps), '--kl-coefficient', str(kl_coefficient)]
    if mode != 'smoke': args += ['--learner-microbatch-size', str(learner_microbatch_size)]
    if resume: args += ['--resume-from', resume]
    if keep_all_checkpoints: args.append('--keep-all-checkpoints')
    if resume_state_retention != 'all': args += ['--resume-state-retention', resume_state_retention]
    if allow_epoch_extension: args.append('--allow-epoch-extension')
    environment = dict(LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE='4',
                       CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    try:
        with comparison_guard as credit_calls, patch.dict(os.environ, environment), patch('sys.argv', args), \
                patch.object(torch.cuda, 'set_device'), \
                patch.object(torch.cuda, 'get_device_name', return_value='CPU test redirect; no CUDA'), \
                patch.object(dist, 'init_process_group'), \
                patch.object(AutoTokenizer, 'from_pretrained', return_value=tok), \
                patch.object(SamplerClient, 'launch', return_value=FixedResponseSampler()), \
                patch.object(trainer, 'load_actor', side_effect=cpu_actor), \
                patch.object(trainer, 'features', side_effect=tracked_reference_features), \
                patch('torch.nn.parallel.DistributedDataParallel', side_effect=cpu_ddp), \
                patch.object(checkpoints, 'capture_rng', side_effect=lambda device=None: capture_rng(None)), \
                patch.object(checkpoints, 'restore_rng', side_effect=lambda state, device=None: restore_rng(state, None)), \
                patch.object(module, 'version', return_value='CPU-test-runtime'):
            module.main()
            if arm == 'grpo' and mode == 'smoke':
                Path(out, f'grpo_smoke_audit_rank_{rank}.json').write_text(json.dumps({
                    'reference_feature_calls': len(feature_calls),
                    'comparison_credit_calls': credit_calls.call_count}))
    except RuntimeError as error:
        if not interrupt or 'Intentional test interruption' not in str(error): raise
        Path(out, f'expected_interrupt_rank_{rank}.json').write_text(json.dumps({'expected_interruption': True}))
    finally:
        if ddp_models and Path(out).is_dir():
            Path(out, f'ddp_bucket_audit_rank_{rank}.json').write_text(json.dumps({
                'has_rebuilt_buckets': ddp_models[0]._has_rebuilt_buckets,
                'logging': ddp_models[0]._get_ddp_logging_data()}))
        if dist.is_initialized(): dist.destroy_process_group()


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('transformers','peft')), 'Install transformers and peft')
class TrainingLoopCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cls.directory = tempfile.TemporaryDirectory(prefix='stride-controller-cpu-')
        cls.root = Path(cls.directory.name)
        cls.model = cls.root/'model'
        torch.manual_seed(7)
        Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                                   head_dim=8,max_position_embeddings=1024)).save_pretrained(cls.model)
        cls.data = cls.root/'data.json'
        cls.data.write_text(json.dumps([dict(id=str(i), question=f'Question {i}: What is 1 + 1?',ground_truth=2) for i in range(4)]))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def launch(self, name, arm='local_positive', mode='train', epochs=1, resume=None, interrupt=False,
               stable_buckets=False, data_path=None, batch_size=4, keep_all_checkpoints=False,
               resume_state_retention='all', allow_epoch_extension=False,
                warmup_steps=0, kl_coefficient=0., learner_microbatch_size=1):
        out = self.root/name
        workers = mp.spawn(loop_worker, args=(str(self.root/(name+'_init')),str(self.model),str(data_path or self.data),
                           str(out),arm,mode,epochs,str(resume) if resume else None,interrupt,stable_buckets,batch_size,
                           keep_all_checkpoints, resume_state_retention, allow_epoch_extension,
                           warmup_steps, kl_coefficient, learner_microbatch_size),
                           nprocs=4,join=False)
        try:
            deadline = time.monotonic() + 60
            while not workers.join(timeout=1):
                if time.monotonic() >= deadline:
                    self.fail('Four-rank CPU integration exceeded its 60-second limit')
        finally:
            for process in workers.processes:
                if process.is_alive(): process.terminate()
            for process in workers.processes:
                process.join(timeout=2)
                if process.is_alive():
                    process.kill(); process.join(timeout=2)
        return out

    def test_keep_all_retains_initial_and_non_epoch_checkpoints(self):
        import resumable_checkpoint as checkpoints
        data = self.root / 'retention_data.json'
        data.write_text(json.dumps([dict(id=str(i), question=f'Retention question {i}: What is 1 + 1?',
                                         ground_truth=2) for i in range(16)]))
        out = self.launch('retained', arm='grpo', data_path=data, keep_all_checkpoints=True)
        started = json.loads((out / 'started.json').read_text())
        done = json.loads((out / 'done.json').read_text())
        self.assertTrue(started['args']['keep_all_checkpoints'])
        self.assertEqual(started['checkpoint_retention'], 'all')
        self.assertEqual(done['checkpoint_retention'], 'all')
        for step in range(5):
            checkpoint = checkpoints.inspect_checkpoint(out / f'state_{step:03d}',
                                                         started['training_contract'], 4)
            self.assertEqual(checkpoint['completed_step'], step)
            self.assertTrue((out / f'adapter_{step:03d}/adapter_model.safetensors').is_file())
            if step:
                saved = json.loads((out / f'checkpoint_{step:03d}.json').read_text())
                self.assertEqual(saved['checkpoint_retention'], 'all')
                self.assertEqual(saved['pruned_steps'], [])

    def test_latest_epoch_state_retention_keeps_all_adapters_and_receipts(self):
        import resumable_checkpoint as checkpoints
        data = self.root / 'latest_epoch_data.json'
        data.write_text(json.dumps([dict(id=str(i), question=f'Latest epoch question {i}: What is 1 + 1?',
                                         ground_truth=2) for i in range(16)]))
        out = self.launch('latest_epoch_states', arm='grpo', data_path=data, keep_all_checkpoints=True,
                          resume_state_retention='latest-epoch')
        self.assertEqual({p.name for p in out.glob('state_*')}, {'state_004'})
        self.assertEqual({p.name for p in out.glob('adapter_*') if p.is_dir()},
                         {f'adapter_{i:03d}' for i in range(5)})
        for step in range(5):
            receipt = checkpoints.inspect_checkpoint_receipt(out, step)
            self.assertFalse(receipt['resumable'])
        checkpoints.inspect_checkpoint(out / 'state_004')
        self.assertEqual(json.loads((out / 'done.json').read_text())['resume_state_retention'], 'latest-epoch')

    def test_explicit_four_to_six_epoch_extension_matches_uninterrupted_training(self):
        from safetensors.torch import load_file
        options = dict(arm='grpo', stable_buckets=True, keep_all_checkpoints=True,
                       resume_state_retention='latest-epoch')
        uninterrupted = self.launch('extension_uninterrupted', epochs=6, **options)
        first = self.launch('extension_first_four', epochs=4, **options)
        immutable = {str(p.relative_to(first)): p.read_bytes() for p in first.rglob('*') if p.is_file()}
        continued = self.launch('extension_to_six', epochs=6, resume=first / 'state_004',
                                allow_epoch_extension=True, **options)
        self.assertEqual(immutable, {str(p.relative_to(first)): p.read_bytes() for p in first.rglob('*') if p.is_file()})
        self.assert_nested_exact(load_file(str(uninterrupted / 'adapter_006/adapter_model.safetensors')),
                                 load_file(str(continued / 'adapter_006/adapter_model.safetensors')))
        for name in ['optimizer.pt'] + [f'rng_rank_{rank}.pt' for rank in range(4)]:
            self.assert_nested_exact(torch.load(uninterrupted / 'state_006' / name, weights_only=True),
                                     torch.load(continued / 'state_006' / name, weights_only=True))
        started = json.loads((continued / 'started.json').read_text())
        self.assertEqual(started['epoch_extension_from'], 4)
        self.assertEqual(started['training_contract']['epochs'], 6)
        for step in (4, 5):
            for rank in range(4):
                # Request/output path provenance varies; the generated token,
                # logprob, score and credit records must remain identical.
                self.assertEqual(json.loads((uninterrupted / f'rank_{rank}/rollouts_{step:03d}.json').read_text()),
                                 json.loads((continued / f'rank_{rank}/rollouts_{step:03d}.json').read_text()))

    def test_four_rank_smoke_and_each_training_arm(self):
        smoke = self.launch('smoke',mode='smoke')
        self.assertTrue(json.loads((smoke/'done.json').read_text())['complete'])
        for rank in range(4):
            report = json.loads((smoke/f'rank_{rank}/smoke_checks.json').read_text())
            self.assertTrue(report['all_rank_trainable_weights_equal'])
            self.assertEqual(report['fixed_reference_maxdiff'], 0)
        for arm in ('local_positive','single_step','token_uniform','grpo'):
            out = self.launch(arm,arm=arm)
            done = json.loads((out/'done.json').read_text())
            start = json.loads((out/'started.json').read_text())
            self.assertEqual((done['steps'],done['start_step']),(1,0))
            self.assertEqual(start['actual_format_coefficient'],0.)
            self.assertEqual(start['attempt_initialization'],'base_model_fresh_lora')
            self.assertTrue((out/'state_001/COMPLETE.json').is_file())
            update = json.loads((out/'update_001.json').read_text())
            self.assertEqual(update['rollout_count'],12)
            self.assertEqual(update['totals']['correct'],8)
            self.assertEqual(update['totals']['gate_active'],4)
            self.assertGreater(update['gradient_norm'],0)
            self.assertLess(update['actor_behavior_abs_logp_mean'],.001)

    def test_partial_batch_training_resume_preserves_real_rows_and_optimizer(self):
        from safetensors.torch import load_file
        data = self.root/'partial_data.json'
        data.write_text(json.dumps([dict(id=str(i), question=f'Partial question {i}: What is 1 + 1?',
                                        ground_truth=2) for i in range(14)]))
        kwargs = dict(arm='grpo', epochs=1, stable_buckets=True, data_path=data, batch_size=8)
        full = self.launch('partial_continuous', **kwargs)
        stopped = self.launch('partial_stopped', interrupt=True, **kwargs)
        resumed = self.launch('partial_resumed', resume=stopped/'state_001', **kwargs)
        expected_ids = [str(i) for i in range(14)]
        visits = []
        for step in range(2):
            update = json.loads((full/f'update_{step+1:03d}.json').read_text())
            self.assertEqual(update['rollout_count'], (8 if step == 0 else 6) * 3)
            self.assertEqual(update['totals']['groups'], 8 if step == 0 else 6)
            self.assertEqual(update['collective_padding_forwards'], 0 if step == 0 else 6)
            for rank in range(4):
                groups = json.loads((full/f'rank_{rank}/rollouts_{step:03d}.json').read_text())
                visits.extend(group['id'] for group in groups)
        self.assertEqual(sorted(visits, key=int), expected_ids)
        self.assertEqual(json.loads((full/'done.json').read_text())['epochs'], 1)
        self.assertEqual(json.loads((resumed/'done.json').read_text())['epochs'], 1)
        self.assertEqual(json.loads((resumed/'started.json').read_text())['start_step'], 1)
        self.assertEqual(json.loads((full/'update_002.json').read_text())['metrics']['train/epoch'], 1)
        self.assert_nested_exact(load_file(str(full/'adapter_002/adapter_model.safetensors')),
                                 load_file(str(resumed/'adapter_002/adapter_model.safetensors')))
        self.assert_nested_exact(torch.load(full/'state_002/optimizer.pt', weights_only=True),
                                 torch.load(resumed/'state_002/optimizer.pt', weights_only=True))

    def test_interruption_resume_matches_uninterrupted_model_and_optimizer(self):
        self.assert_resume_continuity(stable_buckets=False)

    def test_grpo_smoke_keeps_reference_and_adapter_checks_without_comparison_credits(self):
        out = self.launch('grpo_smoke', arm='grpo', mode='smoke')
        self.assertTrue(json.loads((out/'done.json').read_text())['complete'])
        fingerprints = []
        for rank in range(4):
            audit = json.loads((out/f'grpo_smoke_audit_rank_{rank}.json').read_text())
            self.assertEqual(audit['comparison_credit_calls'], 0)
            self.assertEqual(audit['reference_feature_calls'], 2)
            check = json.loads((out/f'rank_{rank}/smoke_checks.json').read_text())
            self.assertEqual(check['fixed_reference_maxdiff'], 0)
            self.assertGreater(check['adapter_changed_logps'], 1e-5)
            self.assertLess(check['fresh_adapter_abs_logp_mean'], .05)
            self.assertTrue(check['all_rank_trainable_weights_equal'])
            fingerprints.append(check['trainable_sha256'])
            groups = json.loads((out/f'rank_{rank}/rollouts_000.json').read_text())
            self.assertTrue(all(group['features'] is None for group in groups))
        self.assertEqual(len(set(fingerprints)), 1)
        update = json.loads((out/'update_001.json').read_text())
        self.assertGreater(update['gradient_norm'], 0)
        self.assertLess(update['actor_behavior_abs_logp_mean'], .05)

    def test_fixed_ddp_buckets_resume_model_and_optimizer_exactly(self):
        self.assert_resume_continuity(stable_buckets=True)

    def test_kl_warmup_pair_real_checkpoint_resume_exact(self):
        self.assert_resume_continuity(stable_buckets=True, warmup_steps=3,
                                      kl_coefficient=.01, learner_microbatch_size=2)

    def assert_nested_exact(self, expected, actual):
        self.assertIs(type(expected), type(actual))
        if isinstance(expected, torch.Tensor):
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected: self.assert_nested_exact(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual): self.assert_nested_exact(left, right)
        else:
            self.assertEqual(expected, actual)

    def assert_resume_tensor(self, expected, actual, *, exact, label):
        self.assertEqual(expected.dtype, torch.float32, label)
        if exact:
            self.assertTrue(torch.equal(expected, actual), label)
            return
        # The fresh reducer and the already-rebuilt reducer pack parameters in
        # different orders. Gloo's ring sums their flat-buffer chunks in rank-
        # dependent order. Bound the resulting FP32 error at each tensor's own
        # scale; this is less than one part per million, with no absolute floor.
        scale = max(float(expected.abs().max()), float(actual.abs().max()))
        torch.testing.assert_close(expected, actual, rtol=0,
                                   atol=8 * torch.finfo(torch.float32).eps * scale,
                                   msg=label)

    def assert_resume_continuity(self, *, stable_buckets, warmup_steps=0, kl_coefficient=0., learner_microbatch_size=1):
        from safetensors.torch import load_file
        prefix = ('stabilized_' if kl_coefficient else '') + ('fixed_' if stable_buckets else '')
        options = dict(epochs=2, stable_buckets=stable_buckets, warmup_steps=warmup_steps,
                       kl_coefficient=kl_coefficient, learner_microbatch_size=learner_microbatch_size)
        full = self.launch(prefix+'continuous', **options)
        stopped = self.launch(prefix+'stopped',interrupt=True, **options)
        self.assertFalse((stopped/'done.json').exists())
        self.assertTrue((stopped/'state_001/COMPLETE.json').is_file())
        self.assertEqual(len(list(stopped.glob('expected_interrupt_rank_*.json'))),4)
        continued = self.launch(prefix+'continued',resume=stopped/'state_001', **options)
        if kl_coefficient:
            from stabilization import learning_rate_at_update
            for update_index in (1, 2):
                audit = json.loads((full/f'update_{update_index:03d}.json').read_text())
                expected_lr = learning_rate_at_update(.01, warmup_steps, update_index - 1)
                self.assertEqual(audit['learning_rate'], expected_lr)
                self.assertEqual(audit['metrics']['train/learning_rate'], expected_lr)
                self.assertEqual(audit['kl_coefficient'], kl_coefficient)
                self.assertGreaterEqual(audit['sampled_kl_mean'], 0.)
            self.assertGreater(audit['sampled_kl_mean'], 0.)
            continued_audit = json.loads((continued/'update_002.json').read_text())
            self.assertEqual(continued_audit['learning_rate'], audit['learning_rate'])
        self.assertEqual(json.loads((continued/'done.json').read_text())['start_step'],1)
        left = load_file(str(full/'adapter_002/adapter_model.safetensors'))
        right = load_file(str(continued/'adapter_002/adapter_model.safetensors'))
        previous = load_file(str(full/'adapter_001/adapter_model.safetensors'))
        stopped_previous = load_file(str(stopped/'adapter_001/adapter_model.safetensors'))
        resumed_previous = load_file(str(continued/'adapter_001/adapter_model.safetensors'))
        self.assert_nested_exact(previous, stopped_previous)
        self.assert_nested_exact(stopped_previous, resumed_previous)
        for name in ['optimizer.pt'] + [f'rng_rank_{rank}.pt' for rank in range(4)]:
            before = torch.load(full/'state_001'/name, weights_only=True)
            saved = torch.load(stopped/'state_001'/name, weights_only=True)
            restored = torch.load(continued/'state_001'/name, weights_only=True)
            self.assert_nested_exact(before, saved)
            self.assert_nested_exact(saved, restored)
        for rank in range(4):
            # Identical serialized groups establish that the resumed update
            # receives exactly the same tokens, log probabilities and credit.
            self.assertEqual(json.loads((full/f'rank_{rank}/rollouts_001.json').read_text()),
                             json.loads((continued/f'rank_{rank}/rollouts_001.json').read_text()))
            for path, rebuilt in [(full, not stable_buckets), (stopped, False), (continued, False)]:
                report = json.loads((path/f'ddp_bucket_audit_rank_{rank}.json').read_text())
                self.assertEqual(report['has_rebuilt_buckets'], rebuilt)
            self.assert_nested_exact(torch.load(full/f'state_002/rng_rank_{rank}.pt', weights_only=True),
                                     torch.load(continued/f'state_002/rng_rank_{rank}.pt', weights_only=True))
        a = torch.load(full/'state_002/optimizer.pt',weights_only=True)
        b = torch.load(continued/'state_002/optimizer.pt',weights_only=True)
        optimizer_diff = {}
        epsilon_error = {}
        for key in a['state']:
            for field, value in a['state'][key].items():
                other = b['state'][key][field]
                difference = float((value-other).abs().max())
                optimizer_diff[field] = max(optimizer_diff.get(field, 0.), difference)
                scale = max(float(value.abs().max()), float(other.abs().max()))
                epsilon_error[field] = max(epsilon_error.get(field, 0.), difference/(torch.finfo(value.dtype).eps*scale) if scale else 0.)
        epsilon_error['adapter'] = max(float((left[k]-right[k]).abs().max()) /
                                      (torch.finfo(left[k].dtype).eps * max(float(left[k].abs().max()), float(right[k].abs().max())))
                                      for k in left)
        print(json.dumps({'resume_numerical_audit': {
            'stable_buckets':stable_buckets,
            'max_abs_model_difference':max(float((left[k]-right[k]).abs().max()) for k in left),
            'max_abs_preinterruption_difference':max(float((previous[k]-stopped_previous[k]).abs().max()) for k in previous),
            'unequal_model_elements':sum(int((left[k]!=right[k]).sum()) for k in left),
            'max_abs_optimizer_difference':optimizer_diff,
            'max_difference_in_tensor_scaled_fp32_epsilons':epsilon_error,
            'ddp_buckets':{name:json.loads((path/'ddp_bucket_audit_rank_0.json').read_text())['has_rebuilt_buckets'] for name,path in [('continuous',full),('stopped',stopped),('continued',continued)]}}}), flush=True)
        self.assertEqual(set(left),set(right))
        for key in left:
            self.assert_resume_tensor(left[key],right[key],exact=stable_buckets,label=key)
        self.assertEqual(a['param_groups'],b['param_groups'])
        for key in a['state']:
            for field,value in a['state'][key].items():
                self.assert_resume_tensor(value,b['state'][key][field],
                                          exact=stable_buckets or field == 'step',label=field)


if __name__ == '__main__':
    unittest.main()
