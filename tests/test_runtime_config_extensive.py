"""Remaining runtime validation boundaries and long-schedule retention checks.

These tests read only tiny local fixtures; they do not initialize GPUs or launch
training. Large epoch counts exercise integer range guards without iterating.
"""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import runtime_config as config
import trainer
import trainer_resumable


def arguments():
    parser = argparse.ArgumentParser()
    config.add_training_arguments(parser, resumable=True)
    return parser.parse_args(['train', '--model-path', '/fixture/model', '--data', '/fixture/data.json',
                              '--data-sha256', 'a'*64, '--out', '/fixture/run'])


class RuntimeConfigurationBoundaryTests(unittest.TestCase):
    def test_training_mode_sampler_limit_and_wandb_requirements(self):
        cases = [('mode', 'warmup', 'Only direct training'),
                 ('sampler_max_num_seqs', 8, 'Sampler sequence limit'),
                 ('wandb_mode', 'online', 'explicit --wandb-project'),
                 ('wandb_mode', 'offline', 'explicit --wandb-project')]
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                args = arguments(); setattr(args, field, value)
                if field == 'wandb_mode':
                    args.wandb_project = None
                with self.assertRaisesRegex(ValueError, message):
                    config.validate_options(args, resumable=True)
        args = arguments(); args.wandb_mode = 'online'; args.wandb_project = 'fixture-project'
        config.validate_options(args, resumable=True)

    def test_smoke_resume_rejected_and_complete_uppercase_hash_normalized(self):
        args = arguments(); args.mode = 'smoke'; args.resume_from = '/fixture/state_001'
        with self.assertRaisesRegex(ValueError, 'Smoke does not support resume'):
            config.validate_options(args, resumable=False)
        args = arguments(); args.data_sha256 = 'ABCDEF01'*8
        config.validate_options(args, resumable=True)
        self.assertEqual(args.data_sha256, 'abcdef01'*8)

    def test_missing_model_configuration_or_weights_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'config.json'):
                config.resolve_model_path(root/'missing')
            with self.assertRaisesRegex(ValueError, 'config.json'):
                config.resolve_model_path(root)
            (root/'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'No local model weights'):
                config.resolve_model_path(root)
            (root/'pytorch_model.bin').write_bytes(b'fixture only; not loadable weights')
            self.assertEqual(config.resolve_model_path(root), root.resolve())
            # A manually prepared model may carry metadata without the optional
            # downloader alias; actual content hashes still establish identity.
            (root/'model_manifest.json').write_text('{"source":"manual fixture"}')
            self.assertEqual(config.resolve_model_path(root, 'qwen4b'), root.resolve())

    def test_row_container_schema_and_id_type_fail_before_gpu(self):
        cases = [None, [], {}, ['invalid'], [{'id': 'one', 'question': 'Q'}],
                 [{'id': True, 'question': 'Q', 'ground_truth': 1}],
                 [{'id': [], 'question': 'Q', 'ground_truth': 1}],
                 [{'id': 'one', 'question': '$\\[\\]$', 'ground_truth': 1}]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                config.validate_rows(rows)

    def test_incomplete_global_batch_and_seed_range_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'training.json'
            rows = [{'id': str(i), 'question': f'Solve problem {i}.', 'ground_truth': i} for i in range(3)]
            path.write_text(json.dumps(rows))
            args = arguments(); args.batch_size = 4; args.data = str(path); args.data_sha256 = config.digest(path)
            with self.assertRaisesRegex(ValueError, 'full global batch'):
                config.load_training_data(args)
            rows.append({'id': '3', 'question': 'Solve problem 3.', 'ground_truth': 3})
            path.write_text(json.dumps(rows)); args.data_sha256 = config.digest(path)
            args.epochs = 2**63
            config.validate_options(args, resumable=True)
            with self.assertRaisesRegex(ValueError, 'sampler seed range'):
                config.load_training_data(args)
            args.epochs = 4
            self.assertEqual(config.load_training_data(args)[1], rows)

    def test_schedule_rejects_short_partial_and_unknown_modes(self):
        args = arguments()
        for size in (0, 32, 65):
            with self.subTest(size=size), self.assertRaises(ValueError):
                config.schedule_steps(args, size)
        args.mode = 'unexpected'
        with self.assertRaisesRegex(ValueError, 'Unknown training mode'):
            config.schedule_steps(args, 64)

    def test_model_identity_includes_template_assets_and_ignores_download_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'config.json').write_text('{}')
            (root/'model.safetensors').write_bytes(b'fixture only')
            (root/'chat_template.jinja').write_text('template one')
            cache = root/'.cache'; cache.mkdir()
            (cache/'metadata.json').write_text('first cache state')
            first = config.model_identity(root)
            (cache/'metadata.json').write_text('second cache state')
            (root/'download.log').write_text('new unsupported ancillary file')
            self.assertEqual(config.model_identity(root), first)
            (root/'chat_template.jinja').write_text('template two')
            self.assertNotEqual(config.model_identity(root)['content_sha256'], first['content_sha256'])

    def test_adapter_origin_accepts_nested_contract_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); adapter = root/'adapter_002'; adapter.mkdir()
            identity = {'content_sha256': 'fixture-base'}
            (root/'started.json').write_text(json.dumps({'training_contract': {'model_identity': identity}}))
            config.validate_adapter_origin(adapter, identity)
            with self.assertRaisesRegex(ValueError, 'different or unverified'):
                config.validate_adapter_origin(adapter, {'content_sha256': 'another-base'})


class LongScheduleLegacyRetentionTests(unittest.TestCase):
    def build_adapters(self, root):
        for step in (998, 1000, 1001, 1004, 1005, 1006):
            path = root/f'adapter_{step:03d}'; path.mkdir()
            (path/'weights.bin').write_bytes(b'fixture')
            (root/f'adapter_{step:03d}_manifest.json').write_text('{}')
        unrelated = root/'adapter_1000_backup'; unrelated.mkdir()
        return unrelated

    def test_preserved_pilot_helper_prunes_four_digit_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); unrelated = self.build_adapters(root)
            trainer.prune_adapters(root, 1005)
            for step in (998, 1000, 1001):
                self.assertFalse((root/f'adapter_{step:03d}').exists())
                self.assertTrue((root/f'adapter_{step:03d}_manifest.json').is_file())
            self.assertTrue(all((root/f'adapter_{step:03d}').is_dir() for step in (1004, 1005, 1006)))
            self.assertTrue(unrelated.is_dir())

    def test_resumable_helper_prunes_four_digits_but_preserves_pinned_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); unrelated = self.build_adapters(root)
            trainer_resumable.prune_synchronization_adapters(root, 1005, {1001}, finetune_mode='lora')
            for step in (998, 1000):
                self.assertFalse((root/f'adapter_{step:03d}').exists())
                self.assertTrue((root/f'adapter_{step:03d}_manifest.json').is_file())
            self.assertTrue(all((root/f'adapter_{step:03d}').is_dir() for step in (1001, 1004, 1005, 1006)))
            self.assertTrue(unrelated.is_dir())


if __name__ == '__main__':
    unittest.main()
