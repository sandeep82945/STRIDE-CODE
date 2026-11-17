"""Offline tests: tiny inert checkpoint bytes, no models, GPUs or Hub calls."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hf_export as export
import resumable_checkpoint as cp
import runtime_config


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root, mode='full', epochs=4, arm='local_positive', *, version=5,
            training_questions=8, batch_size=4, world_size=1):
    base = root / 'base'
    base.mkdir()
    write(base / 'config.json', {'model_type': 'qwen3', '_name_or_path': '/private/original/model'})
    write(base / 'tokenizer_config.json', {'tokenizer_class': 'PreTrainedTokenizerFast',
                                           'name_or_path': '/private/tokenizer'})
    write(base / 'tokenizer.json', {'version': '1.0', 'model': {'type': 'WordLevel'}})
    (base / 'model.safetensors').write_bytes(b'base checkpoint fixture')
    (base / 'chat_template.jinja').write_text('{{ messages }}')
    manifest = dict(status='download_complete', alias='qwen4b',
                    repo_id=export.BASE_MODELS['qwen4b'][0],
                    revision=export.BASE_MODELS['qwen4b'][1],
                    files=[dict(file=p.name, sha256=cp.digest(p), bytes=p.stat().st_size)
                           for p in sorted(base.iterdir())])
    write(base / 'model_manifest.json', manifest)
    identity = runtime_config.model_identity(base)
    lora = dict(rank=16, alpha=32, dropout=0., targets=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                                     'gate_proj', 'up_proj', 'down_proj'])
    contract = dict(version=version, mode='train', model='qwen4b', arm=arm,
                    finetune_mode=mode, warmup_steps=0, format_coefficient=0.,
                    context_limit=8192, epoch_offset=0, diversity_alpha_schedule='constant',
                    training_questions=training_questions, batch_size=batch_size,
                    epochs=epochs, world_size=world_size,
                    seed=42, K=8, alpha=1., data_sha256='a' * 64,
                    model_identity=identity, lora=lora if mode == 'lora' else None)
    per_epoch = (training_questions + batch_size - 1) // batch_size
    if version == 6:
        contract.update(batch_schedule='epoch_shuffle_keep_partial',
                        collective_padding='zero_loss_no_generated_samples',
                        updates_per_epoch=per_epoch)
    run = root / 'run'
    run.mkdir()
    write(run / 'started.json', dict(training_contract=contract, model_identity=identity,
             data_sha256=contract['data_sha256'], start_step=0, resume_from=None,
             args=dict(mode='train', model_path=str(base.resolve()), private_note='DO NOT EXPORT')))
    for epoch in range(1, epochs + 1):
        make_checkpoint(run, epoch * per_epoch, contract, base)
    write(run / 'done.json', dict(complete=True, steps=epochs * per_epoch, epochs=epochs,
                                  finetune_mode=mode))
    return run, base, contract


def make_checkpoint(run, step, contract, base):
    mode = contract['finetune_mode']
    policy = run / f'{"model" if mode == "full" else "adapter"}_{step:03d}'
    policy.mkdir()
    if mode == 'full':
        write(policy / 'config.json', {'model_type': 'qwen3', '_name_or_path': str(base),
                                       'nested': {'_name_or_path': '/private/other'}})
        write(policy / 'tokenizer_config.json', {'tokenizer_class': 'PreTrainedTokenizerFast',
                                                 'name_or_path': str(base)})
        (policy / 'tokenizer.json').write_bytes((base / 'tokenizer.json').read_bytes())
        (policy / 'model.safetensors').write_bytes(b'full checkpoint ' + str(step).encode())
    else:
        lora = contract['lora']
        write(policy / 'adapter_config.json', dict(peft_type='LORA', task_type='CAUSAL_LM',
             r=lora['rank'], lora_alpha=lora['alpha'], lora_dropout=lora['dropout'],
             target_modules=lora['targets'], base_model_name_or_path=str(base)))
        (policy / 'adapter_model.safetensors').write_bytes(b'adapter ' + str(step).encode())
        (policy / 'README.md').write_text('Original generated card with /private/path; should be replaced')
    cp.write_json(run / (policy.name + '_manifest.json'), {p.name: cp.digest(p) for p in policy.iterdir()})
    stage = cp.prepare_state(run, step)
    for name in cp._state_names(contract['world_size'], mode):
        (stage / name).write_bytes(b'never deserialize this optimizer/RNG fixture')
    return cp.commit_state(run, step, contract['world_size'], contract, mode)


def rewrite_metadata(state, callback):
    metadata = json.loads((state / 'metadata.json').read_text())
    callback(metadata)
    write(state / 'metadata.json', metadata)
    marker = json.loads((state / 'COMPLETE.json').read_text())
    marker['metadata_sha256'] = cp.digest(state / 'metadata.json')
    write(state / 'COMPLETE.json', marker)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)

    def test_pins_match_downloader(self):
        path = Path(__file__).resolve().parents[1] / 'scripts' / 'download_model.py'
        spec = importlib.util.spec_from_file_location('hf_export_test_downloader', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(export.BASE_MODELS,
                         {key: (value['repo_id'], value['revision']) for key, value in module.MODELS.items()})

    def test_full_export_all_selected_epochs_is_portable_and_inference_only(self):
        run, base, contract = fixture(self.root)
        before = (run / 'model_004' / 'config.json').read_bytes()
        results = export.prepare_exports(run, self.root / 'exports', description='My epoch checkpoint.')
        self.assertEqual([item['epoch'] for item in results], [2, 3, 4])
        self.assertEqual([item['step'] for item in results], [4, 6, 8])
        self.assertEqual((run / 'model_004' / 'config.json').read_bytes(), before)
        for item in results:
            folder = Path(item['folder'])
            self.assertTrue(folder.is_absolute())
            manifest = json.loads((folder / 'export_manifest.json').read_text())
            self.assertEqual(manifest, {key: value for key, value in item.items() if key != 'folder'})
            self.assertEqual(set(item['files']), {p.name for p in folder.iterdir()} - {'export_manifest.json'})
            self.assertEqual(item['files'], {name: cp.digest(folder / name) for name in item['files']})
            config = json.loads((folder / 'config.json').read_text())
            self.assertNotIn('_name_or_path', config)
            self.assertEqual(config['nested'], {})
            for path in folder.iterdir():
                if path.suffix != '.safetensors':
                    self.assertNotIn(str(self.root), path.read_text())
                    self.assertNotIn('DO NOT EXPORT', path.read_text())
            self.assertIn('full-parameter training', (folder / 'README.md').read_text())
            self.assertIn('My epoch checkpoint.', (folder / 'README.md').read_text())
            self.assertFalse(any('optimizer' in name or 'rng' in name for name in item['files']))

    def test_lora_base_pin_tokenizer_and_original_bytes_preserved(self):
        run, base, contract = fixture(self.root, 'lora')
        source = run / 'adapter_004' / 'adapter_config.json'
        before = source.read_bytes()
        item = export.prepare_exports(run, run / 'hf_exports', epochs=(2,))[0]
        folder = Path(item['folder'])
        config = json.loads((folder / 'adapter_config.json').read_text())
        self.assertEqual(config['base_model_name_or_path'], item['base_model'])
        self.assertEqual(config['revision'], item['base_revision'])
        self.assertEqual(config['r'], 16)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual((folder / 'tokenizer.json').read_bytes(), (base / 'tokenizer.json').read_bytes())
        self.assertIn('chat_template.jinja', item['files'])
        self.assertNotIn('name_or_path', json.loads((folder / 'tokenizer_config.json').read_text()))
        self.assertIn('AutoPeftModelForCausalLM', (folder / 'README.md').read_text())
        self.assertNotIn('Original generated card', (folder / 'README.md').read_text())

    def test_partial_150_question_schedule_exports_real_epoch_boundaries(self):
        run, _, _ = fixture(self.root, 'lora', version=6, training_questions=150,
                            batch_size=64, world_size=4)
        exported = export.prepare_exports(run, run / 'hf_exports', epochs=(1, 2, 3, 4))
        self.assertEqual([item['step'] for item in exported], [3, 6, 9, 12])
        self.assertEqual([item['epoch'] for item in exported], [1, 2, 3, 4])
        self.assertTrue(all(item['training_questions'] == 150 for item in exported))
        self.assertTrue(all(item['batch_size'] == 64 for item in exported))
        card = (Path(exported[-1]['folder']) / 'README.md').read_text()
        self.assertIn('Nominal global prompt batch', card)
        self.assertIn('22 real questions', card)
        write(run / 'done.json', dict(complete=True, steps=8, epochs=4, finetune_mode='lora'))
        with self.assertRaisesRegex(ValueError, 'successfully completed'):
            export.prepare_exports(run, self.root / 'wrong-schedule')

    def test_partial_schedule_metadata_and_legacy_divisibility_are_required(self):
        _, _, contract = fixture(self.root, 'lora', version=6, training_questions=150,
                                  batch_size=64, world_size=4)
        changes = [dict(version=5), dict(version=7), dict(version=True),
                   dict(updates_per_epoch=2), dict(updates_per_epoch=True),
                   dict(batch_schedule='drop_last'), dict(collective_padding='duplicate_samples'),
                   dict(training_questions=130), dict(training_questions=32, updates_per_epoch=1)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                export._validate_contract({**contract, **change})

    def test_export_never_hashes_or_deserializes_optimizer_rng(self):
        run, _, _ = fixture(self.root)
        original_digest = cp.digest
        def restricted_digest(path):
            self.assertFalse(Path(path).name.endswith('.pt'))
            return original_digest(path)
        with patch.object(cp, 'digest', side_effect=restricted_digest):
            export.prepare_exports(run, self.root / 'exports')

    def test_idempotent_retry_and_payload_tampering_rejected(self):
        run, _, _ = fixture(self.root)
        first = export.prepare_exports(run, self.root / 'exports')
        self.assertEqual(first, export.prepare_exports(run, self.root / 'exports'))
        (Path(first[0]['folder']) / 'README.md').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            export.prepare_exports(run, self.root / 'exports')

    def test_retry_cannot_change_description_or_extra_payload(self):
        run, _, _ = fixture(self.root)
        item = export.prepare_exports(run, self.root / 'exports', epochs=(2,))[0]
        with self.assertRaisesRegex(ValueError, 'differs'):
            export.prepare_exports(run, self.root / 'exports', epochs=(2,), description='Changed')
        extra = Path(item['folder']) / '.env'
        extra.write_text('secret')
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            export.prepare_exports(run, self.root / 'exports', epochs=(2,))
        self.assertTrue(extra.exists())

    def test_default_selection_filters_epochs_beyond_schedule(self):
        run, _, _ = fixture(self.root, epochs=2)
        self.assertEqual([item['epoch'] for item in export.prepare_exports(run, self.root / 'exports')], [2])
        with self.assertRaisesRegex(ValueError, 'No requested'):
            export.prepare_exports(run, self.root / 'empty', epochs=(3, 4))
        self.assertFalse((self.root / 'empty').exists())

    def test_epoch_validation(self):
        run, _, _ = fixture(self.root)
        for epochs in ((), (0,), (True,), (1.0,), (2, 2), '2,3', None):
            with self.subTest(epochs=epochs), self.assertRaisesRegex(ValueError, 'epochs'):
                export.prepare_exports(run, self.root / 'exports', epochs=epochs)

    def test_incomplete_done_or_missing_commit_or_tampered_metadata_rejected(self):
        run, _, _ = fixture(self.root)
        done = run / 'done.json'
        original = done.read_bytes()
        value = json.loads(original); value['complete'] = False; write(done, value)
        with self.assertRaisesRegex(ValueError, 'successfully completed'):
            export.prepare_exports(run, self.root / 'exports')
        done.write_bytes(original)
        marker = run / 'state_004' / 'COMPLETE.json'; marker.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or unsafe'):
            export.prepare_exports(run, self.root / 'exports')
        marker.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'metadata hash'):
            export.prepare_exports(run, self.root / 'exports')

    def test_policy_tamper_and_committed_provenance_change_rejected(self):
        run, _, _ = fixture(self.root)
        weights = run / 'model_004' / 'model.safetensors'
        before = weights.read_bytes(); weights.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'Policy hash'):
            export.prepare_exports(run, self.root / 'exports')
        weights.write_bytes(before)
        rewrite_metadata(run / 'state_004', lambda metadata: metadata['policy'].update(manifest_sha256='b' * 64))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            export.prepare_exports(run, self.root / 'exports')

    def test_missing_or_extra_optimizer_state_rejected_without_deletion(self):
        run, _, _ = fixture(self.root)
        extra = run / 'state_004' / 'unexpected.txt'; extra.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            export.prepare_exports(run, self.root / 'exports')
        self.assertTrue(extra.exists()); extra.unlink()
        state = run / 'state_004' / 'rng_rank_0.pt'; state.unlink()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            export.prepare_exports(run, self.root / 'exports')

    def test_unexpected_policy_member_rejected_even_if_manifest_and_commit_match(self):
        run, _, contract = fixture(self.root)
        policy = run / 'model_004'
        (policy / '.env').write_text('NEVER UPLOAD')
        manifest = {p.name: cp.digest(p) for p in policy.iterdir()}
        write(run / 'model_004_manifest.json', manifest)
        provenance = cp.verify_policy(policy, 'full')
        rewrite_metadata(run / 'state_004', lambda meta: meta.update(policy=provenance))
        with self.assertRaisesRegex(ValueError, 'Unexpected full-model'):
            export.prepare_exports(run, self.root / 'exports')

    def test_symlink_policy_or_tokenizer_or_destination_rejected(self):
        run, base, _ = fixture(self.root, 'lora')
        destination = self.root / 'export_link'; destination.symlink_to(base, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            export.prepare_exports(run, destination)
        tokenizer = base / 'tokenizer.json'; payload = tokenizer.read_bytes(); tokenizer.unlink()
        other = self.root / 'other.json'; other.write_bytes(payload); tokenizer.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            export.prepare_exports(run, self.root / 'exports')

    def test_tokenizer_changes_and_fake_model_label_rejected(self):
        run, base, _ = fixture(self.root, 'lora')
        tok = base / 'tokenizer.json'; original = tok.read_bytes(); tok.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'tokenizer changed'):
            export.prepare_exports(run, self.root / 'exports')
        tok.write_bytes(original)
        manifest = base / 'model_manifest.json'; data = json.loads(manifest.read_text())
        data['repo_id'] = 'another/model'; write(manifest, data)
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            export.prepare_exports(run, self.root / 'exports')

    def test_credentials_in_description_rejected_and_not_written(self):
        run, _, _ = fixture(self.root)
        with patch.dict(os.environ, {'HF_TOKEN': 'hf_SECRET', 'WANDB_API_KEY': 'wandb_SECRET'}):
            for description in ('key is hf_SECRET', 'key is wandb_SECRET'):
                with self.assertRaisesRegex(ValueError, 'Credentials'):
                    export.prepare_exports(run, self.root / 'exports', description=description)
        self.assertFalse((self.root / 'exports').exists())

    def test_overlap_rejected_and_no_source_deleted(self):
        run, base, _ = fixture(self.root)
        for destination in (run, base, run / 'model_004', run / 'state_004' / 'export'):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, 'overlap'):
                export.prepare_exports(run, destination)
        self.assertTrue((run / 'model_004' / 'model.safetensors').is_file())

    def test_copy_fallback_and_failed_copy_do_not_publish_or_delete_source(self):
        run, _, _ = fixture(self.root)
        with patch.object(export.os, 'link', side_effect=OSError('Cross-device')):
            item = export.prepare_exports(run, self.root / 'exports', epochs=(2,))[0]
        self.assertTrue((Path(item['folder']) / 'model.safetensors').is_file())
        with patch.object(export, '_transfer', side_effect=OSError('copy interrupted')):
            with self.assertRaisesRegex(OSError, 'copy interrupted'):
                export.prepare_exports(run, self.root / 'failed', epochs=(2,))
        self.assertEqual(list((self.root / 'failed').iterdir()), [])
        self.assertTrue((run / 'model_004' / 'model.safetensors').is_file())

    def test_resume_ancestry_recovers_earlier_epoch(self):
        run, base, contract = fixture(self.root)
        resumed = self.root / 'resumed'; resumed.mkdir()
        started = json.loads((run / 'started.json').read_text())
        started.update(start_step=4, resume_from=str(run / 'state_004'))
        write(resumed / 'started.json', started)
        make_checkpoint(resumed, 6, contract, base)
        make_checkpoint(resumed, 8, contract, base)
        write(resumed / 'done.json', json.loads((run / 'done.json').read_text()))
        result = export.prepare_exports(resumed, self.root / 'exports')
        self.assertEqual([item['step'] for item in result], [4, 6, 8])
        self.assertEqual(result[0]['run_fingerprint'], cp.digest(resumed / 'started.json'))

    def test_unknown_arm_or_mismatched_state_contract_rejected(self):
        run, _, _ = fixture(self.root)
        path = run / 'started.json'; original = path.read_bytes()
        started = json.loads(original); started['training_contract']['arm'] = 'unknown'; write(path, started)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            export.prepare_exports(run, self.root / 'exports')
        path.write_bytes(original)
        rewrite_metadata(run / 'state_004', lambda meta: meta['contract'].update(alpha=2.))
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            export.prepare_exports(run, self.root / 'exports')

    def test_missing_pruned_epoch_fails_without_removing_other_checkpoints(self):
        run, _, _ = fixture(self.root)
        state = run / 'state_004'; state.rename(run / 'saved_state')
        with self.assertRaisesRegex(ValueError, 'no retained'):
            export.prepare_exports(run, self.root / 'exports')
        self.assertTrue((run / 'saved_state').is_dir())
        self.assertTrue((run / 'state_006').is_dir())

    def test_real_tiny_full_and_lora_exports_load_without_network(self):
        """Load actual random tiny Qwen3 weights/tokenizer; no pretrained assets."""
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, PeftModel, get_peft_model
        for mode in ('full', 'lora'):
            with self.subTest(mode=mode):
                local = self.root / mode; local.mkdir()
                run, base, contract = fixture(local, mode, epochs=2)
                tiny = Qwen3ForCausalLM(Qwen3Config(vocab_size=4, hidden_size=16,
                    intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=2, head_dim=8))
                tiny.save_pretrained(base, safe_serialization=True)
                tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
                    WordLevel({'<unk>': 0, '<eos>': 1, 'a': 2, 'b': 3}, unk_token='<unk>')),
                    unk_token='<unk>', eos_token='<eos>')
                tokenizer.save_pretrained(base)
                manifest = dict(status='download_complete', alias='qwen4b',
                    repo_id=export.BASE_MODELS['qwen4b'][0], revision=export.BASE_MODELS['qwen4b'][1],
                    files=[dict(file=p.name, sha256=cp.digest(p), bytes=p.stat().st_size)
                           for p in sorted(base.iterdir()) if p.name != 'model_manifest.json'])
                write(base / 'model_manifest.json', manifest)
                identity = runtime_config.model_identity(base)
                contract['model_identity'] = identity
                started = json.loads((run / 'started.json').read_text())
                started.update(model_identity=identity, training_contract=contract)
                write(run / 'started.json', started)
                # Replace fixture policy files before committing the tiny real pair.
                import shutil
                for epoch in (1, 2):
                    step = epoch * 2
                    policy = run / f'{"model" if mode == "full" else "adapter"}_{step:03d}'
                    shutil.rmtree(policy); policy.mkdir()
                    if mode == 'full':
                        tiny.save_pretrained(policy, safe_serialization=True)
                        tokenizer.save_pretrained(policy)
                    else:
                        tiny.config._name_or_path = str(base)
                        actor = get_peft_model(Qwen3ForCausalLM(tiny.config), LoraConfig(
                            r=16, lora_alpha=32, lora_dropout=0., task_type='CAUSAL_LM',
                            target_modules=contract['lora']['targets']))
                        actor.save_pretrained(policy, safe_serialization=True)
                    write(run / (policy.name + '_manifest.json'), {p.name: cp.digest(p) for p in policy.iterdir()})
                    provenance = cp.verify_policy(policy, mode)
                    def update(metadata):
                        metadata.update(contract=contract, policy=provenance)
                        if mode == 'lora':
                            metadata['adapter'] = provenance
                    rewrite_metadata(run / f'state_{step:03d}', update)
                with patch.dict(os.environ, {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}):
                    item = export.prepare_exports(run, local / 'exports', epochs=(2,))[0]
                    loaded_tokenizer = AutoTokenizer.from_pretrained(item['folder'], local_files_only=True)
                    if mode == 'full':
                        loaded = AutoModelForCausalLM.from_pretrained(item['folder'], local_files_only=True)
                    else:
                        loaded = PeftModel.from_pretrained(tiny, item['folder'], local_files_only=True,
                                                           torch_device='cpu')
                    with torch.no_grad():
                        output = loaded(input_ids=torch.tensor([[2, 3]]))
                    self.assertEqual(tuple(output.logits.shape), (1, 2, 4))
                    self.assertEqual(loaded_tokenizer.convert_tokens_to_ids('a'), 2)


if __name__ == '__main__':
    unittest.main()
