"""CPU publisher tests; all Hub calls use an in-memory fake service."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import hf_publish


class HttpFailure(RuntimeError):
    def __init__(self, code):
        super().__init__('untrusted service response includes hf_secret_for_test_only')
        self.response = SimpleNamespace(status_code=code)


class FakeHub:
    endpoint = 'https://huggingface.co'

    def __init__(self, root):
        self.root = Path(root)
        self.repos = {}
        self.calls = []
        self.identity = {'name': 'alice', 'orgs': [{'name': 'lab'}],
                         'auth': {'accessToken': {'role': 'write'}}}
        self.fail_before_upload = []
        self.fail_after_upload = []
        self.corrupt_remote = False
        self.privacy_after_upload = None
        self.before_upload = None

    def whoami(self, token):
        self.calls.append(('whoami', {}))
        self.last_token = token
        return self.identity

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(('create_repo', dict(repo_id=repo_id, **{k: v for k, v in kwargs.items() if k != 'token'})))
        if repo_id not in self.repos:
            self.repos[repo_id] = {'private': kwargs['private'], 'sha': 'initial',
                                   'files': {'.gitattributes': b'*.safetensors filter=lfs\n'}}
        return repo_id

    def repo_info(self, repo_id, **kwargs):
        self.calls.append(('repo_info', dict(repo_id=repo_id, **{k: v for k, v in kwargs.items() if k != 'token'})))
        repo = self.repos[repo_id]
        if kwargs.get('revision') and kwargs['revision'] != repo['sha']:
            raise HttpFailure(404)
        siblings = []
        for name, data in repo['files'].items():
            lfs = {'sha256': hashlib.sha256(data).hexdigest()} if name.endswith('.safetensors') else None
            if lfs and self.corrupt_remote:
                lfs['sha256'] = 'f' * 64
            siblings.append(SimpleNamespace(rfilename=name, size=len(data), lfs=lfs,
                blob_id=hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()))
        return SimpleNamespace(sha=repo['sha'], private=repo['private'], siblings=siblings)

    def upload_folder(self, **kwargs):
        self.calls.append(('upload_folder', {k: v for k, v in kwargs.items() if k != 'token'}))
        if self.fail_before_upload:
            raise self.fail_before_upload.pop(0)
        if self.before_upload:
            self.before_upload(kwargs)
        repo = self.repos[kwargs['repo_id']]
        if repo['sha'] != kwargs['parent_commit']:
            raise HttpFailure(409)
        for name in kwargs['allow_patterns']:
            repo['files'][name] = (Path(kwargs['folder_path']) / name).read_bytes()
        repo['sha'] = 'a' * 40
        if self.privacy_after_upload is not None:
            repo['private'] = self.privacy_after_upload
        if self.fail_after_upload:
            raise self.fail_after_upload.pop(0)
        return SimpleNamespace(oid=repo['sha'])

    def hf_hub_download(self, repo_id, filename, **kwargs):
        self.calls.append(('hf_hub_download', dict(repo_id=repo_id, filename=filename,
                          **{k: v for k, v in kwargs.items() if k != 'token'})))
        repo = self.repos[repo_id]
        if kwargs['revision'] != repo['sha']:
            raise AssertionError('Expected a pinned remote manifest download')
        path = self.root / 'remote_manifest.json'
        path.write_bytes(repo['files'][filename])
        return str(path)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.run = self.root / 'run'
        self.run.mkdir()
        self.api = FakeHub(self.root)
        env = patch.dict(os.environ, {'HF_TOKEN': 'hf_secret_for_test_only'}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.exports = [self.export()]
        self.sleeps = []

    def export(self, *, arm='local_positive', mode='full', epoch=2):
        folder = self.run / 'hf_exports' / f'epoch-{epoch:03}'
        folder.mkdir(parents=True, exist_ok=True)
        if mode == 'full':
            files = {'config.json': b'{"model_type":"qwen3"}', 'model.safetensors': b'fake full weights'}
        else:
            files = {'adapter_config.json': b'{"r":16}', 'adapter_model.safetensors': b'fake adapter weights'}
        files.update({'README.md': b'# Test checkpoint\n', 'tokenizer_config.json': b'{}',
                      'tokenizer.json': b'{"test":"tokenizer"}'})
        for name, data in files.items():
            (folder / name).write_bytes(data)
        item = dict(model='qwen4b', arm=arm, finetune_mode=mode, seed=42, K=8,
                    alpha=1.0, epoch=epoch, step=epoch * 32, training_questions=2048,
                    batch_size=64, total_epochs=4, dataset_sha256='d' * 64,
                    run_fingerprint='a' * 64, base_model='Qwen/Qwen3-4B-Instruct-2507',
                    base_revision='1234567890', folder=str(folder),
                    files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
        self.write_manifest(item)
        return item

    @staticmethod
    def write_manifest(item):
        manifest = {key: value for key, value in item.items() if key != 'folder'}
        (Path(item['folder']) / 'export_manifest.json').write_text(json.dumps(manifest, sort_keys=True))

    def upload(self, **kwargs):
        def export_fn(run_dir, destination, epochs, description):
            self.assertEqual(run_dir, self.run)
            self.assertEqual(destination, self.run / 'hf_exports')
            return self.exports
        options = dict(namespace='alice', api=self.api, export_fn=export_fn, sleep_fn=self.sleeps.append)
        options.update(kwargs)
        return hf_publish.upload_run(self.run, **options)

    def calls(self, name):
        return [options for called, options in self.api.calls if called == name]

    def test_preflight_is_read_only_and_accepts_user_and_member_organization(self):
        for namespace in ('alice', 'lab'):
            result = hf_publish.preflight_auth(namespace, api=self.api)
            self.assertEqual(result['namespace'], namespace)
            self.assertEqual(result['authenticated_user'], 'alice')
        self.assertEqual([name for name, _ in self.api.calls], ['whoami', 'whoami'])
        self.assertEqual(self.api.last_token, 'hf_secret_for_test_only')

    def test_preflight_refuses_unrelated_namespace_and_read_token(self):
        with self.assertRaisesRegex(hf_publish.PublishError, 'match the token owner'):
            hf_publish.preflight_auth('someone-else', api=self.api)
        self.api.identity['auth']['accessToken']['role'] = 'read'
        with self.assertRaisesRegex(hf_publish.PublishError, 'read-only'):
            hf_publish.preflight_auth('alice', api=self.api)
        self.assertFalse(self.calls('create_repo'))

    def test_preflight_rejects_missing_key_and_noncanonical_endpoint(self):
        with patch.dict(os.environ, {'HF_TOKEN': ''}), self.assertRaisesRegex(hf_publish.PublishError, 'Set HF_TOKEN'):
            hf_publish.preflight_auth('alice', api=self.api)
        self.api.endpoint = 'https://example.invalid'
        with self.assertRaisesRegex(hf_publish.PublishError, 'official Hugging Face endpoint'):
            hf_publish.preflight_auth('alice', api=self.api)
        self.assertFalse(self.api.calls)

    def test_upload_private_full_model_and_verifies_every_file_at_commit(self):
        result = self.upload()
        self.assertTrue(result['complete'], result)
        checkpoint = result['checkpoints'][0]
        self.assertEqual(checkpoint['status'], 'uploaded')
        self.assertTrue(checkpoint['private'])
        self.assertEqual(checkpoint['commit'], 'a' * 40)
        self.assertIn('/commit/' + 'a' * 40, checkpoint['commit_url'])
        create = self.calls('create_repo')[0]
        self.assertTrue(create['private'])
        self.assertTrue(create['exist_ok'])
        upload = self.calls('upload_folder')[0]
        self.assertEqual(upload['parent_commit'], 'initial')
        self.assertEqual(upload['path_in_repo'], '')
        self.assertEqual(set(upload['allow_patterns']), set(self.exports[0]['files']) | {'export_manifest.json'})
        self.assertFalse(upload['run_as_future'])
        self.assertEqual(self.calls('hf_hub_download')[-1]['revision'], checkpoint['commit'])
        self.assertEqual(json.loads((self.run / 'hf_upload_status.json').read_text()), result)
        self.assertTrue((Path(self.exports[0]['folder']) / 'model.safetensors').exists())

    def test_all_stride_variants_private_and_grpo_public_in_both_modes(self):
        for mode in ('full', 'lora'):
            for arm in ('local_positive', 'single_step', 'token_uniform', 'grpo'):
                with self.subTest(mode=mode, arm=arm):
                    shutil.rmtree(self.run / 'hf_exports')
                    self.exports = [self.export(arm=arm, mode=mode)]
                    self.api = FakeHub(self.root)
                    result = self.upload()
                    self.assertTrue(result['complete'], result)
                    self.assertEqual(self.calls('create_repo')[0]['private'], arm != 'grpo')

    def test_private_pilot_override_keeps_every_method_private(self):
        with patch.dict(os.environ, {'HF_PRIVATE_ALL': 'true'}):
            for arm in ('local_positive', 'single_step', 'token_uniform', 'grpo'):
                with self.subTest(arm=arm):
                    shutil.rmtree(self.run / 'hf_exports')
                    self.exports = [self.export(arm=arm, mode='lora')]
                    self.api = FakeHub(self.root)
                    result = self.upload()
                    self.assertTrue(result['complete'], result)
                    self.assertTrue(result['private_all'])
                    self.assertTrue(result['checkpoints'][0]['private'])
                    self.assertTrue(self.calls('create_repo')[0]['private'])

    def test_invalid_pilot_privacy_fails_before_network_and_cannot_flip_existing_repo(self):
        with patch.dict(os.environ, {'HF_PRIVATE_ALL': 'yes'}):
            result = self.upload()
        self.assertFalse(result['complete'])
        self.assertIn('HF_PRIVATE_ALL', result['error'])
        self.assertFalse(self.api.calls)
        shutil.rmtree(self.run / 'hf_exports')
        self.exports = [self.export(arm='grpo', mode='lora')]
        self.assertTrue(self.upload()['complete'])
        self.api.calls.clear()
        with patch.dict(os.environ, {'HF_PRIVATE_ALL': 'true'}):
            result = self.upload()
        self.assertFalse(result['complete'])
        self.assertIn('visibility', result['checkpoints'][0]['error'])
        self.assertFalse(self.calls('upload_folder'))
        self.assertTrue(all(not repo['private'] for repo in self.api.repos.values()))

    def test_epochs_have_distinct_names_and_each_is_loadable_at_repository_root(self):
        self.exports += [self.export(epoch=3), self.export(epoch=4)]
        result = self.upload()
        self.assertTrue(result['complete'], result)
        repos = [checkpoint['repo_id'] for checkpoint in result['checkpoints']]
        self.assertEqual(len(set(repos)), 3)
        for epoch, repo in zip((2, 3, 4), repos):
            self.assertIn(f'-e{epoch:02}-', repo)
            self.assertIn('config.json', self.api.repos[repo]['files'])
            self.assertIn('model.safetensors', self.api.repos[repo]['files'])

    def test_repository_name_contains_settings_and_is_stable_unique_and_bounded(self):
        first = hf_publish.repository_id('lab', 'stride-iclr2027', self.exports[0])
        self.assertIn('qwen3-4b-instruct-2507-local-positive-full-s42-k8-a1-e02-', first)
        self.assertEqual(first, hf_publish.repository_id('lab', 'stride-iclr2027', self.exports[0]))
        changed = dict(self.exports[0], run_fingerprint='b' * 64)
        self.assertNotEqual(first, hf_publish.repository_id('lab', 'stride-iclr2027', changed))
        long = dict(changed, base_model='org/' + 'a-very-long-model-name-' * 4)
        name = hf_publish.repository_id('lab', 'a-very-long-prefix-' * 10, long).split('/')[1]
        self.assertLessEqual(len(name), 96)
        self.assertIn('local-positive-full-s42-k8-a1-e02-', name)
        self.assertNotIn('--', name)

    def test_second_upload_verifies_and_does_not_commit_again(self):
        first = self.upload()
        second = self.upload()
        self.assertTrue(second['complete'], second)
        self.assertEqual(second['checkpoints'][0]['status'], 'already_uploaded')
        self.assertEqual(first['checkpoints'][0]['commit'], second['checkpoints'][0]['commit'])
        self.assertEqual(len(self.calls('upload_folder')), 1)

    def test_lost_upload_response_recovers_by_verifying_remote_manifest(self):
        self.api.fail_after_upload.append(ConnectionError('lost reply hf_secret_for_test_only'))
        result = self.upload()
        self.assertTrue(result['complete'], result)
        self.assertEqual(result['checkpoints'][0]['status'], 'already_uploaded')
        self.assertEqual(result['checkpoints'][0]['attempts'], 2)
        self.assertNotIn('error', result['checkpoints'][0])
        self.assertEqual(len(self.calls('upload_folder')), 1)
        self.assertEqual(self.sleeps, [1])

    def test_transient_failures_retry_with_bound_and_permanent_failure_does_not(self):
        self.api.fail_before_upload = [HttpFailure(503), HttpFailure(429)]
        result = self.upload()
        self.assertTrue(result['complete'], result)
        self.assertEqual(result['checkpoints'][0]['attempts'], 3)
        self.assertEqual(self.sleeps, [1, 2])
        self.api = FakeHub(self.root)
        self.api.fail_before_upload = [HttpFailure(403), HttpFailure(503)]
        result = self.upload()
        self.assertFalse(result['complete'])
        self.assertEqual(result['checkpoints'][0]['attempts'], 1)
        self.assertIn('HTTP 403', result['checkpoints'][0]['error'])
        self.assertNotIn('hf_secret_for_test_only', json.dumps(result))

    def test_retry_exhaustion_is_recorded_and_later_epochs_still_publish(self):
        self.exports.append(self.export(epoch=3))
        self.api.fail_before_upload = [HttpFailure(503)] * 3
        result = self.upload()
        self.assertFalse(result['complete'])
        self.assertEqual([item['status'] for item in result['checkpoints']], ['failed', 'uploaded'])
        self.assertEqual(result['checkpoints'][0]['attempts'], 3)

    def test_existing_wrong_visibility_is_refused_without_flipping(self):
        repo = hf_publish.repository_id('alice', 'stride-iclr2027', self.exports[0])
        self.api.repos[repo] = dict(private=False, sha='initial', files={})
        result = self.upload()
        self.assertFalse(result['complete'])
        self.assertIn('visibility', result['checkpoints'][0]['error'])
        self.assertFalse(self.api.repos[repo]['private'])
        self.assertFalse(self.calls('upload_folder'))

    def test_existing_unrelated_repository_is_never_overwritten(self):
        repo = hf_publish.repository_id('alice', 'stride-iclr2027', self.exports[0])
        self.api.repos[repo] = dict(private=True, sha='initial', files={'README.md': b'another model'})
        result = self.upload()
        self.assertFalse(result['complete'])
        self.assertIn('unrelated files', result['checkpoints'][0]['error'])
        self.assertFalse(self.calls('upload_folder'))

    def test_existing_manifest_mismatch_is_not_overwritten(self):
        result = self.upload()
        repo = self.api.repos[result['checkpoints'][0]['repo_id']]
        manifest = json.loads(repo['files']['export_manifest.json'])
        manifest['epoch'] = 99
        repo['files']['export_manifest.json'] = json.dumps(manifest).encode()
        result = self.upload()
        self.assertFalse(result['complete'])
        self.assertIn('different checkpoint', result['checkpoints'][0]['error'])
        self.assertEqual(len(self.calls('upload_folder')), 1)

    def test_remote_hash_size_missing_and_extra_files_fail_verification(self):
        for mutation in ('hash', 'size', 'missing', 'extra'):
            with self.subTest(mutation=mutation):
                self.api = FakeHub(self.root)
                first = self.upload()
                repo = self.api.repos[first['checkpoints'][0]['repo_id']]
                if mutation == 'hash':
                    self.api.corrupt_remote = True
                elif mutation == 'size':
                    repo['files']['model.safetensors'] += b'x'
                elif mutation == 'missing':
                    del repo['files']['README.md']
                else:
                    repo['files']['optimizer.pt'] = b'secret state'
                result = self.upload()
                self.assertFalse(result['complete'])
                self.assertEqual(len(self.calls('upload_folder')), 1)

    def test_wrong_privacy_after_commit_does_not_report_success(self):
        self.api.privacy_after_upload = False
        result = self.upload()
        self.assertFalse(result['complete'])
        self.assertIn('visibility', result['checkpoints'][0]['error'])

    def test_missing_parent_commit_refuses_unguarded_upload(self):
        repo = hf_publish.repository_id('alice', 'stride-iclr2027', self.exports[0])
        self.api.repos[repo] = dict(private=True, sha=None, files={})
        result = self.upload()
        self.assertFalse(result['complete'])
        self.assertIn('commit', result['checkpoints'][0]['error'])
        self.assertFalse(self.calls('upload_folder'))

    def test_concurrent_branch_change_cannot_overwrite_other_content(self):
        def race(kwargs):
            repo = self.api.repos[kwargs['repo_id']]
            repo.update(sha='concurrent')
            repo['files']['README.md'] = b'concurrent unrelated edit'
        self.api.before_upload = race
        result = self.upload()
        self.assertFalse(result['complete'])
        self.assertIn('unrelated', result['checkpoints'][0]['error'])
        self.assertEqual(len(self.calls('upload_folder')), 1)

    def test_changed_local_file_fails_before_network(self):
        (Path(self.exports[0]['folder']) / 'model.safetensors').write_bytes(b'changed')
        result = self.upload()
        self.assertFalse(result['complete'])
        self.assertIn('changed', result['error'])
        self.assertFalse(self.api.calls)

    def test_extra_hidden_optimizer_or_symlink_files_fail_before_network(self):
        folder = Path(self.exports[0]['folder'])
        for name in ('.env', 'optimizer.pt', 'rollouts.jsonl', 'link.json'):
            with self.subTest(name=name):
                path = folder / name
                if name == 'link.json':
                    path.symlink_to(folder / 'config.json')
                else:
                    path.write_text('not public')
                result = self.upload()
                self.assertFalse(result['complete'])
                self.assertFalse(self.api.calls)
                path.unlink()

    def test_declaring_an_unsafe_file_in_manifest_cannot_expand_allowlist(self):
        item = self.exports[0]
        path = Path(item['folder']) / 'optimizer.pt'
        path.write_bytes(b'optimizer state')
        item['files']['optimizer.pt'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_manifest(item)
        result = self.upload()
        self.assertFalse(result['complete'])
        self.assertIn('allowlist', result['error'])
        self.assertFalse(self.api.calls)

    def test_secret_in_card_or_metadata_never_reaches_network_or_status(self):
        item = self.exports[0]
        path = Path(item['folder']) / 'README.md'
        path.write_text('HF_TOKEN=hf_secret_for_test_only')
        item['files']['README.md'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_manifest(item)
        result = self.upload()
        self.assertFalse(result['complete'])
        self.assertNotIn('hf_secret_for_test_only', json.dumps(result))
        self.assertFalse(self.api.calls)

    def test_path_escape_invalid_arm_and_missing_model_are_rejected(self):
        item = self.exports[0]
        for overrides in ({'folder': str(self.root)}, {'arm': 'unknown'}, {'seed': -1}, {'K': 0}):
            with self.subTest(overrides=overrides):
                self.exports = [dict(item, **overrides)]
                result = self.upload()
                self.assertFalse(result['complete'])
                self.assertFalse(self.api.calls)
        self.exports = [item]
        del item['files']['model.safetensors']
        (Path(item['folder']) / 'model.safetensors').unlink()
        self.write_manifest(item)
        result = self.upload()
        self.assertFalse(result['complete'])
        self.assertIn('missing required', result['error'])

    def test_no_exports_unrequested_or_duplicate_epoch_fail_before_network(self):
        item = self.exports[0]
        for items in ([], [item, item], [dict(item, epoch=1)]):
            with self.subTest(items=len(items)):
                self.exports = items
                result = self.upload()
                self.assertFalse(result['complete'])
                self.assertFalse(self.api.calls)

    def test_input_validation_and_secret_names(self):
        for options in ({'namespace': '../alice'}, {'epochs': ()}, {'epochs': (2, 2)},
                        {'retries': 0}, {'retries': 11}, {'prefix': 'hf_secret_for_test_only'}):
            with self.subTest(options=options), self.assertRaises(hf_publish.PublishError):
                self.upload(**options)
        self.assertFalse(self.api.calls)

    def test_cli_preflight_needs_no_run_and_removes_only_child_offline_setting(self):
        output = io.StringIO()
        with patch.dict(os.environ, {'HF_HUB_OFFLINE': '1', 'HF_ENDPOINT': 'https://example.invalid',
                                    'TRANSFORMERS_OFFLINE': '1'}), \
             patch.object(hf_publish, 'preflight_auth', return_value={'namespace': 'alice'}) as preflight, \
             redirect_stdout(output):
            self.assertEqual(hf_publish.main(['--preflight', '--namespace', 'alice']), 0)
            self.assertNotIn('HF_HUB_OFFLINE', os.environ)
            self.assertEqual(os.environ['HF_ENDPOINT'], hf_publish.HUB_ENDPOINT)
            self.assertEqual(os.environ['TRANSFORMERS_OFFLINE'], '1')
        preflight.assert_called_once_with('alice')
        self.assertNotIn('hf_secret_for_test_only', output.getvalue())

    def test_cli_upload_exit_status_and_sanitized_errors(self):
        for complete in (True, False):
            with patch.object(hf_publish, 'upload_run', return_value={'complete': complete}), redirect_stdout(io.StringIO()):
                self.assertEqual(hf_publish.main(['--run-dir', str(self.run), '--namespace', 'alice']), 0 if complete else 1)
        output = io.StringIO()
        with patch.object(hf_publish, 'preflight_auth', side_effect=RuntimeError('hf_secret_for_test_only')), \
             redirect_stderr(output):
            self.assertEqual(hf_publish.main(['--preflight', '--namespace', 'alice']), 1)
        self.assertNotIn('hf_secret_for_test_only', output.getvalue())

    def test_epoch_parser_rejects_invalid_lists(self):
        self.assertEqual(hf_publish._epochs('2, 3,4'), (2, 3, 4))
        for value in ('', '0,2', '2,2', 'x', '1.5', '-1'):
            with self.subTest(value=value), self.assertRaises(Exception):
                hf_publish._epochs(value)

    def test_real_exporter_integrates_with_publisher_full_and_lora(self):
        from tests.test_hf_export import fixture
        import hf_export
        self.assertEqual(hf_publish.TOKENIZER_FILES, hf_export.TOKENIZER_FILES)
        for mode in ('full', 'lora'):
            with self.subTest(mode=mode):
                folder = self.root / ('real-' + mode)
                folder.mkdir()
                run, _, _ = fixture(folder, mode=mode)
                api = FakeHub(folder)
                result = hf_publish.upload_run(run, namespace='alice', api=api,
                    description='Integration fixture; no trained model.', sleep_fn=self.sleeps.append)
                self.assertTrue(result['complete'], result)
                self.assertEqual(len(result['checkpoints']), 3)
                for checkpoint in result['checkpoints']:
                    files = api.repos[checkpoint['repo_id']]['files']
                    self.assertIn('training_metadata.json', files)
                    self.assertIn('export_manifest.json', files)
                    self.assertIn('tokenizer.json', files)
                    self.assertFalse(any('optimizer' in name or 'state_' in name for name in files))
                again = hf_publish.upload_run(run, namespace='alice', api=api,
                    description='Integration fixture; no trained model.', sleep_fn=self.sleeps.append)
                self.assertTrue(again['complete'], again)
                self.assertEqual([item['status'] for item in again['checkpoints']], ['already_uploaded'] * 3)


if __name__ == '__main__':
    unittest.main()
