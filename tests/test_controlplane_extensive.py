"""Local control-plane integration tests; fake GPU reports are explicitly mocked.

Tiny parquet fixtures and synthetic model asset bytes never download weights or
launch training. Archive/launcher tests run real filesystem operations.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import torch
import run
import format_credit

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location('controlplane_' + name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data = load_script('prepare_data')
models = load_script('download_model')
release = load_script('package_release')
runtime = load_script('check_runtime')


@contextmanager
def command_line(*args):
    output, error = io.StringIO(), io.StringIO()
    with patch.object(sys, 'argv', ['test-command', *map(str, args)]), redirect_stdout(output), redirect_stderr(error):
        yield output, error


def raw_rows(count=12):
    return [dict(prompt=f'Compute {i} times 7.', solution=str(i*7),
                 reward_model=dict(ground_truth=str(i*7)), extra_info=dict(index=i))
            for i in range(count)]


def fake_snapshot(**kwargs):
    root = Path(kwargs['local_dir'])
    for name, value in {'config.json': b'{}', 'tokenizer_config.json': b'{}',
                        'model.safetensors': b'fixture, not a model'}.items():
        (root/name).write_bytes(value)


class SourceAndLauncherTests(unittest.TestCase):
    def test_concurrent_source_freezing_produces_one_complete_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, cache = Path(tmp)/'source', Path(tmp)/'cache'
            (source/'nested').mkdir(parents=True)
            (source/'a.py').write_text('x=1\n')
            (source/'nested'/'b.py').write_text('y=2\n')
            with ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(lambda _: run.freeze_source(source, cache), range(8)))
            self.assertEqual(len(set(paths)), 1)
            self.assertEqual(list(cache.iterdir()), [paths[0]])
            manifest = json.loads((paths[0]/'source_manifest.json').read_text())
            for name, expected in manifest.items():
                self.assertEqual(hashlib.sha256((paths[0]/name).read_bytes()).hexdigest(), expected)

    def test_source_snapshot_rejects_symlinks_empty_source_and_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source in (root/'missing', root/'empty'):
                if source.name == 'empty': source.mkdir()
                with self.assertRaises(ValueError):
                    run.freeze_source(source, root/'cache')
            source = root/'source'; source.mkdir()
            (root/'outside.py').write_text('x=1')
            (source/'a.py').symlink_to(root/'outside.py')
            with self.assertRaisesRegex(ValueError, 'symlinked'):
                run.freeze_source(source, root/'cache')
            (source/'a.py').unlink(); (source/'a.py').write_text('x=1')
            snapshot = run.freeze_source(source, root/'cache')
            (snapshot/'source_manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'manifest'):
                run.freeze_source(source, root/'cache')

    def test_main_dry_run_has_no_gpu_or_snapshot_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.json'; path.write_text('[]')
            with command_line('eval', '--dry-run', '--data', path, '--model-path', '/fixture/model') as (out, _), \
                    patch.object(run, 'freeze_source') as freeze, patch.object(run, 'execution_environment') as env, \
                    patch.object(run.subprocess, 'call') as child:
                self.assertEqual(run.main(), 0)
            report = json.loads(out.getvalue())
            self.assertEqual(report['required_gpus'], 1)
            self.assertTrue(report['dry_run'])
            freeze.assert_not_called(); env.assert_not_called(); child.assert_not_called()

    def test_main_uses_frozen_source_and_propagates_child_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.json'; path.write_text('[]')
            source = Path(tmp)/'frozen'
            with command_line('train', '--data', path, '--model-path', '/fixture/model', '--wandb-mode', 'disabled', '--hf-upload', 'off'), \
                    patch.object(run, 'freeze_source', return_value=source), \
                    patch.object(run, 'execution_environment', return_value={'allocation': 'kept'}), \
                    patch.object(run.subprocess, 'call', return_value=17) as child:
                self.assertEqual(run.main(), 17)
            self.assertIn(str(source/'trainer_resumable.py'), child.call_args.args[0])
            self.assertEqual(child.call_args.kwargs['env'], {'allocation': 'kept'})

    def test_main_configuration_error_exits_before_subprocess(self):
        with command_line('train', '--model-path', '/fixture/model'), patch.object(run.subprocess, 'call') as child:
            with self.assertRaises(SystemExit) as caught:
                run.main()
        self.assertEqual(caught.exception.code, 2)
        child.assert_not_called()

    def test_environment_gpu_selection_and_offline_defaults(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(torch.version, 'hip', None), \
                patch.object(torch.cuda, 'device_count', return_value=8):
            result = run.execution_environment('train')
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
        self.assertEqual(result['HF_HUB_OFFLINE'], '1')
        with patch.dict(os.environ, {}, clear=True), patch.object(torch.version, 'hip', 'mock-rocm'):
            with self.assertRaisesRegex(ValueError, 'ROCm'):
                run.execution_environment('train')
        for visible in ('0,0,1,2,3,4,5,6', '0,1,2,3,4,5,6,', ''):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': visible}, clear=True):
                with self.assertRaises(ValueError): run.execution_environment('train')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-fixture', 'WANDB_MODE': 'online'}, clear=True):
            self.assertEqual(run.execution_environment('eval')['WANDB_MODE'], 'online')

    def test_option_and_build_command_reject_missing_values(self):
        for arguments in (['--data'], ['--data', '--out', 'x']):
            with self.assertRaisesRegex(ValueError, 'Missing value'):
                run.option(arguments, '--data')
        self.assertIsNone(run.option(['--out', 'x'], '--data'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.json'; path.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'model-path'):
                run.build_command('train', ['--data', str(path)])


class DataCommandTests(unittest.TestCase):
    def parquet(self, root):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = root/'tiny-source.parquet'
        pq.write_table(pa.Table.from_pylist(raw_rows()), path)
        return path

    def test_real_tiny_parquet_main_hashes_exclusions_and_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = self.parquet(root)
            excluded = root/'exclude.jsonl'
            excluded.write_text('{"source_id":"0"}\n{"question":"Compute 1 times 7."}\n')
            results = []
            for suffix in ('first', 'second'):
                out = root/suffix
                with command_line('--source-parquet', source, '--out', out, '--train-count', 4,
                                  '--val-count', 2, '--test-count', 2, '--exclude-json', excluded) as (stdout, _), \
                        patch.object(data, 'SOURCE_SHA256', data.digest(source)):
                    data.main()
                report = json.loads(stdout.getvalue())
                self.assertEqual(report['eligible_pool'], 10)
                selected = []
                for split, metadata in report['splits'].items():
                    self.assertEqual(metadata['sha256'], data.digest(out/f'{split}.json'))
                    selected.extend(json.loads((out/f'{split}.json').read_text()))
                self.assertFalse({'0', '1'} & {r['id'] for r in selected})
                self.assertEqual(len(selected), 8)
                results.append(report['splits'])
            self.assertEqual(results[0], results[1])

    def test_download_branch_is_pinned_using_mock_download_and_real_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = self.parquet(root)
            with command_line('--out', root/'out', '--train-count', 2, '--val-count', 1, '--test-count', 1), \
                    patch.object(data, 'SOURCE_SHA256', data.digest(source)), \
                    patch('huggingface_hub.hf_hub_download', return_value=str(source)) as download:
                data.main()
            download.assert_called_once_with(repo_id=data.DATASET, repo_type='dataset',
                                             filename=data.SOURCE_FILENAME, revision=data.REVISION)

    def test_main_rejects_bad_source_existing_output_and_empty_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = self.parquet(root)
            with command_line('--source-parquet', source, '--out', root/'out'):
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'): data.main()
            self.assertFalse((root/'out').exists())
            with command_line('--out', root):
                with self.assertRaises(SystemExit): data.main()
            with command_line('--out', root/'out', '--train-count', 0):
                with self.assertRaises(SystemExit): data.main()
            excluded = root/'unsupported.json'; excluded.write_text('{"id":"0"}')
            with command_line('--source-parquet', source, '--out', root/'out', '--exclude-json', excluded), \
                    patch.object(data, 'SOURCE_SHA256', data.digest(source)):
                with self.assertRaisesRegex(ValueError, 'No supported exclusion'): data.main()
            self.assertFalse((root/'out').exists())

    def test_source_metadata_and_encoding_fail_closed(self):
        for row in ({'prompt': 'q'}, {'prompt': 'q', 'reward_model': None}):
            with self.assertRaisesRegex(ValueError, 'metadata'): data.source_records([row])
        with self.assertRaises(ValueError): data.encode({'not-json': float('nan')})


class ModelCommandTests(unittest.TestCase):
    def test_download_main_and_verified_resume_use_no_real_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'model'
            with command_line('--model', 'qwen17b', '--out', out) as (stdout, _), \
                    patch('huggingface_hub.snapshot_download', side_effect=fake_snapshot) as snapshot:
                models.main()
            self.assertEqual(json.loads(stdout.getvalue())['status'], 'download_complete')
            self.assertEqual(snapshot.call_args.kwargs['revision'], models.MODELS['qwen17b']['revision'])
            with command_line('--model', 'qwen17b', '--out', out, '--resume') as (stdout, _), \
                    patch('huggingface_hub.snapshot_download') as snapshot:
                models.main()
            snapshot.assert_not_called()
            self.assertEqual(json.loads(stdout.getvalue())['status'], 'verified_existing_download')

    def test_inventory_rejects_symlink_assets_and_missing_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/'model'; out.mkdir()
            with self.assertRaisesRegex(ValueError, 'missing'): models.inventory(out)
            fake_snapshot(local_dir=out)
            (root/'external').write_bytes(b'outside')
            (out/'asset.json').symlink_to(root/'external')
            with self.assertRaisesRegex(ValueError, 'regular files'): models.inventory(out)
            (out/'asset.json').unlink()
            (out/'.model_manifest.stale.tmp').write_text('ignored staging file')
            self.assertEqual(len(models.inventory(out)), 3)

    def test_model_metadata_and_output_symlinks_are_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); actual = root/'metadata'; actual.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'metadata object'): models.read_regular_json(actual)
            link = root/'link'; link.symlink_to(actual)
            with self.assertRaisesRegex(ValueError, 'regular metadata'): models.read_regular_json(link)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                models.download_model('qwen4b', link, snapshot_fn=fake_snapshot)

    def test_manifest_publication_race_does_not_overwrite_other_completion(self):
        real_link = os.link
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'model'
            def racing_link(source, destination):
                Path(destination).write_bytes(b'concurrent completion')
                return real_link(source, destination)
            with patch.object(models.os, 'link', side_effect=racing_link):
                with self.assertRaises(FileExistsError):
                    models.download_model('qwen4b', out, snapshot_fn=fake_snapshot)
            self.assertEqual((out/models.MANIFEST_FILE).read_bytes(), b'concurrent completion')
            self.assertFalse(list(out.glob('.model_manifest.*.tmp')))


class ReleaseTests(unittest.TestCase):
    def release_tree(self, root):
        for name in release.TOP | {'scripts/prepare_data.py', 'scripts/download_model.py', 'docs/VALIDATION.md', 'training/model.py'}:
            path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture source\n')
        return root

    def test_release_contains_only_sources_and_manifest_hashes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.release_tree(Path(tmp)/'source'); target = Path(tmp)/'release.zip'
            for name in ('data/train.json', 'models/config.json', 'runs/result.json', 'training/__pycache__/secret.py', 'docs/.private.txt'):
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('must not ship')
            (root/'training'/'link.py').symlink_to(root/'training'/'model.py')
            with patch.object(release, 'ROOT', root), command_line('--out', target) as (stdout, _):
                release.main()
            self.assertTrue(json.loads(stdout.getvalue())['verified'])
            with zipfile.ZipFile(target) as archive:
                manifest = json.loads(archive.read('stride_handoff/RELEASE_MANIFEST.json'))
                for name, expected in manifest['source_files'].items():
                    self.assertEqual(hashlib.sha256(archive.read('stride_handoff/'+name)).hexdigest(), expected)
                self.assertFalse(any('must not ship' in archive.read(name).decode() for name in archive.namelist()))
                self.assertNotIn('stride_handoff/training/link.py', archive.namelist())
            before = target.read_bytes()
            with patch.object(release, 'ROOT', root), command_line('--out', target):
                with self.assertRaises(FileExistsError): release.main()
            self.assertEqual(target.read_bytes(), before)

    def test_archive_write_failure_leaves_no_published_or_staged_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.release_tree(Path(tmp)/'source'); target = Path(tmp)/'release.zip'
            with patch.object(release, 'ROOT', root), command_line('--out', target), \
                    patch.object(zipfile.ZipFile, 'writestr', side_effect=OSError('simulated disk failure')):
                with self.assertRaisesRegex(OSError, 'disk failure'): release.main()
            self.assertFalse(target.exists())
            self.assertFalse(list(target.parent.glob('.stride-release-*')))

    def test_archive_validation_failure_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.release_tree(Path(tmp)/'source'); target = Path(tmp)/'release.zip'
            with patch.object(release, 'ROOT', root), command_line('--out', target), \
                    patch.object(zipfile.ZipFile, 'testzip', return_value='damaged member'):
                with self.assertRaisesRegex(ValueError, 'CRC'): release.main()
            self.assertFalse(target.exists())
            self.assertFalse(list(target.parent.glob('.stride-release-*')))

    def test_publish_race_preserves_other_archive(self):
        real_link = os.link
        with tempfile.TemporaryDirectory() as tmp:
            root = self.release_tree(Path(tmp)/'source'); target = Path(tmp)/'release.zip'
            def competing_publish(source, destination):
                Path(destination).write_bytes(b'other publisher')
                return real_link(source, destination)
            with patch.object(release, 'ROOT', root), command_line('--out', target), \
                    patch.object(release.os, 'link', side_effect=competing_publish):
                with self.assertRaises(FileExistsError): release.main()
            self.assertEqual(target.read_bytes(), b'other publisher')
            self.assertFalse(list(target.parent.glob('.stride-release-*')))

    def test_missing_required_source_fails_before_release_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.release_tree(Path(tmp)/'source'); target = Path(tmp)/'release.zip'
            (root/'docs'/'VALIDATION.md').unlink()
            with patch.object(release, 'ROOT', root), command_line('--out', target):
                with self.assertRaisesRegex(ValueError, 'Missing release inputs'): release.main()
            self.assertFalse(target.exists())


def mocked_imports(torch_module=None):
    """Dependency import doubles; never evidence of actual CUDA compatibility."""
    values = {}
    for name, attrs in {
        'transformers': ('AutoTokenizer', 'AutoModelForCausalLM'),
        'peft': ('LoraConfig', 'PeftModel'), 'vllm': ('LLM', 'SamplingParams'),
        'vllm.lora.request': ('LoRARequest',),
        'vllm.config.compilation': ('CompilationConfig', 'CompilationMode', 'CUDAGraphMode'),
    }.items():
        module = types.ModuleType(name)
        for attr in attrs: setattr(module, attr, object)
        values[name] = module
    if torch_module is not None: values['torch'] = torch_module
    return values


class RuntimeProbeTests(unittest.TestCase):
    def test_runtime_version_error_is_machine_readable_and_nonzero(self):
        with command_line('--gpus', 1) as (out, _), patch.object(runtime, 'version', return_value='0.0'):
            status = runtime.main()
        report = json.loads(out.getvalue())
        self.assertEqual(status, 1); self.assertFalse(report['passed'])
        self.assertFalse(report['model_training_tested'])
        self.assertIn('Expected torch', report['error'])

    @unittest.skipIf(torch.cuda.is_available(), 'This assertion requires an actual CPU-only runtime')
    def test_real_cpu_runtime_is_rejected_even_with_import_and_version_doubles(self):
        with command_line('--gpus', 1) as (out, _), \
                patch.object(runtime, 'version', side_effect=runtime.PACKAGES.__getitem__), \
                patch.dict(sys.modules, mocked_imports()):
            status = runtime.main()
        self.assertEqual(status, 1)
        self.assertIn('Expected NVIDIA CUDA', json.loads(out.getvalue())['error'])

    def test_mocked_cuda_probe_success_and_wrong_count_paths(self):
        fake = types.ModuleType('torch')
        fake.version = types.SimpleNamespace(cuda='mock-only', hip=None)
        fake.float32, fake.bfloat16 = torch.float32, torch.bfloat16
        fake.arange = lambda *args, **kwargs: torch.arange(*args, **{**kwargs, 'device': 'cpu'})
        fake.eye = lambda *args, **kwargs: torch.eye(*args, **{**kwargs, 'device': 'cpu'})
        fake.linalg, fake.isfinite = torch.linalg, torch.isfinite
        fake.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
            synchronize=lambda index: None,
            get_device_properties=lambda index: types.SimpleNamespace(name='MOCK NOT GPU', total_memory=1024**3))
        with command_line('--gpus', 1) as (out, _), \
                patch.object(runtime, 'version', side_effect=runtime.PACKAGES.__getitem__), \
                patch.dict(sys.modules, mocked_imports(fake)):
            self.assertEqual(runtime.main(), 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report['devices'][0]['name'], 'MOCK NOT GPU')
        self.assertFalse(report['model_training_tested'])
        fake.cuda.device_count = lambda: 8
        with command_line('--gpus', 1) as (out, _), \
                patch.object(runtime, 'version', side_effect=runtime.PACKAGES.__getitem__), \
                patch.dict(sys.modules, mocked_imports(fake)):
            self.assertEqual(runtime.main(), 1)
        self.assertIn('allocated visible GPUs', json.loads(out.getvalue())['error'])


class DormantFormatReferenceTests(unittest.TestCase):
    def test_correctness_ranking_tie_handling_and_masking(self):
        mask = torch.tensor([[True, True, False], [True, True, True], [True, False, False], [True, True, False]])
        for coefficient in (0., .1, .9):
            output = format_credit.compute_format_credit([1, 1, 0, 0], [0, 1, 1, 0], mask, format_coefficient=coefficient)
            self.assertGreater(output['diagnostics']['min_correct_minus_max_wrong_reward'], 0)
            self.assertFalse(bool(output['advantages'][~mask].any()))
            self.assertTrue(torch.allclose(output['advantages']-output['main'], output['format_effect']))
        tied = format_credit.compute_format_credit([1]*4, [1]*4, mask)
        self.assertFalse(bool(tied['advantages'].any()))
        self.assertEqual(tied['diagnostics']['joint_reward_std'], 0.)
        rescued = format_credit.compute_format_credit([0]*4, [0, 1, 0, 1], mask)
        self.assertTrue(rescued['diagnostics']['format_rescues_accuracy_tie'])

    def test_format_reference_rejects_invalid_inputs(self):
        for coefficient in (True, None, -1., 1., float('inf'), 'invalid'):
            with self.assertRaises(ValueError):
                format_credit.compute_format_credit([1, 0], [1, 0], torch.ones(2, 3, dtype=torch.bool), format_coefficient=coefficient)
        for mask in (None, torch.ones(2, 3), torch.ones(1, 3, dtype=torch.bool), torch.zeros(2, 3, dtype=torch.bool)):
            with self.assertRaises(ValueError): format_credit.compute_format_credit([1, 0], [1, 0], mask)
        for correct, flags in (([1], [1, 0]), ([1, .5], [1, 0]), ([1, 0], [float('nan'), 0])):
            with self.assertRaises(ValueError):
                format_credit.compute_format_credit(correct, flags, torch.ones(2, 3, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
