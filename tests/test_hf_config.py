"""CPU-only Hub upload config tests; no real credentials or network calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import env_config
import hf_config
import run_settings


class HuggingFaceConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'settings.toml'
        self.path.write_text('')
        self.env = {'HF_USERNAME': 'fixture-team', 'HF_TOKEN': 'hf_fixture_not_real'}

    def resolve(self, *, stage='train', contents='', overrides=None, env=None, require=True, settings=True):
        self.path.write_text(contents)
        loaded = run_settings.load_settings(self.path) if settings else None
        return hf_config.resolve_hf_config(stage, loaded, overrides, self.env if env is None else env,
                                            require_credentials=require)

    def test_enabled_defaults_only_return_nonsecret_upload_plan(self):
        result = self.resolve()
        self.assertEqual(result, dict(enabled=True, epochs=[2, 3, 4], namespace='fixture-team',
                                      repo_prefix='stride-iclr2027', description='', retries=3,
                                      private_all=False))
        self.assertNotIn(self.env['HF_TOKEN'], json.dumps(result))

    def test_file_overrides_defaults_and_does_not_mutate_them(self):
        result = self.resolve(contents='[huggingface]\nenabled=true\nepochs=[1, 4]\n'
                                       'repo_prefix="my-study"\ndescription="Reproduction study"\nretries=2')
        self.assertEqual(result['epochs'], [1, 4])
        self.assertEqual(result['repo_prefix'], 'my-study')
        self.assertEqual(result['description'], 'Reproduction study')
        self.assertEqual(result['retries'], 2)
        result['epochs'].append(5)
        self.assertEqual(self.resolve()['epochs'], [2, 3, 4])

    def test_cli_over_env_over_file(self):
        contents = '[huggingface]\nenabled=false\nepochs=[2]\nrepo_prefix="from-file"\nretries=2'
        env = {**self.env, 'HF_UPLOAD_ENABLED': 'true'}
        result = self.resolve(contents=contents, env=env)
        self.assertTrue(result['enabled'])
        result = self.resolve(contents=contents, env=env,
                              overrides=dict(hf_upload='off', hf_epochs='1,3', hf_repo_prefix='from-cli',
                                             hf_description='Run description', hf_retries='5'))
        self.assertFalse(result['enabled'])
        self.assertEqual(result['epochs'], [1, 3])
        self.assertEqual(result['repo_prefix'], 'from-cli')
        self.assertEqual(result['description'], 'Run description')
        self.assertEqual(result['retries'], 5)
        self.assertEqual(result['namespace'], '')

    def test_dotenv_merges_below_exports_and_above_file(self):
        envpath = self.root / '.env'
        envpath.write_text('HF_USERNAME=from-file\nHF_TOKEN=hf_fixture_not_real\nHF_UPLOAD_ENABLED=true\n')
        env = {'HF_USERNAME': 'from-shell', 'HF_UPLOAD_ENABLED': 'false'}
        env_config.load_env_file(envpath, environ=env)
        self.assertFalse(self.resolve(env=env)['enabled'])
        self.assertEqual(self.resolve(env=env, overrides={'hf_upload': 'on'})['namespace'], 'from-shell')

    def test_disabled_does_not_require_credentials(self):
        self.assertFalse(self.resolve(env={}, contents='[huggingface]\nenabled=false')['enabled'])
        self.assertFalse(self.resolve(env={}, overrides={'hf_upload': 'off'})['enabled'])

    def test_no_settings_requires_explicit_enablement(self):
        self.assertFalse(self.resolve(env={}, settings=False)['enabled'])
        self.assertTrue(self.resolve(settings=False, overrides={'hf_upload': 'on'})['enabled'])
        self.assertTrue(self.resolve(settings=False, env={**self.env, 'HF_UPLOAD_ENABLED': 'on'})['enabled'])

    def test_nontraining_stages_force_off_even_with_cli_enablement(self):
        for stage in ('smoke', 'eval'):
            with self.subTest(stage=stage):
                result = self.resolve(stage=stage, env={'HF_UPLOAD_ENABLED': 'true'}, overrides={'hf_upload': 'on'})
                self.assertFalse(result['enabled'])
                self.assertEqual(result['namespace'], '')

    def test_dry_run_can_describe_plan_without_credentials(self):
        result = self.resolve(env={}, require=False)
        self.assertTrue(result['enabled'])
        self.assertEqual(result['namespace'], '')
        self.assertEqual(result['epochs'], [2, 3, 4])

    def test_missing_or_malformed_credentials_fail_before_upload(self):
        for env in ({}, {'HF_TOKEN': 'hf_fixture_not_real'}, {'HF_USERNAME': 'fixture-team'},
                    {**self.env, 'HF_TOKEN': 'short'}, {**self.env, 'HF_TOKEN': 'paste_your_key'},
                    {**self.env, 'HF_TOKEN': 'white space token'}, {**self.env, 'HF_USERNAME': 'https://wrong/name'}):
            with self.subTest(env=env), self.assertRaises(ValueError) as caught:
                self.resolve(env=env)
            self.assertNotIn('hf_fixture_not_real', str(caught.exception))

    def test_config_rejects_embedded_hub_or_wandb_secret_without_echo(self):
        for secret_key in ('HF_TOKEN', 'WANDB_API_KEY'):
            env = {**self.env, secret_key: 'fixture_super_secret'}
            for flag in ('hf_repo_prefix', 'hf_description'):
                with self.subTest(secret_key=secret_key, flag=flag), self.assertRaises(ValueError) as caught:
                    self.resolve(env=env, overrides={flag: 'prefix-fixture_super_secret'})
                self.assertNotIn('fixture_super_secret', str(caught.exception))
        with self.assertRaisesRegex(ValueError, 'Credentials'):
            self.resolve(env={**self.env, 'HF_USERNAME': self.env['HF_TOKEN']})

    def test_invalid_file_types_or_values_rejected(self):
        snippets = ['enabled="true"', 'enabled=1', 'epochs=[]', 'epochs=[true, 3]',
                    'epochs=[2.0, 3.0]', 'epochs=[3, 2]', 'epochs=[2, 2]', 'epochs=[0, 3]',
                    'epochs="2,3,4"', 'repo_prefix="../bad"', 'repo_prefix="bad--prefix"',
                    'repo_prefix=""', 'description=2', 'retries=true', 'retries=0', 'retries=11',
                    'token="fixture_secret"']
        for snippet in snippets:
            with self.subTest(snippet=snippet), self.assertRaises(ValueError) as caught:
                self.resolve(contents='[huggingface]\n' + snippet)
            self.assertNotIn('fixture_secret', str(caught.exception))

    def test_invalid_cli_types_and_values_rejected(self):
        examples = [('hf_upload', 'true'), ('hf_epochs', ''), ('hf_epochs', '2, 3'), ('hf_epochs', '3,2'),
                    ('hf_epochs', '2,2'), ('hf_epochs', '0,3'), ('hf_epochs', [2, 3]),
                    ('hf_retries', '0'), ('hf_retries', '11'), ('hf_retries', '1.0'),
                    ('hf_description', 'line\nbreak'), ('hf_description', 'x' * 4001)]
        for name, value in examples:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.resolve(overrides={name: value})

    def test_environment_boolean_values_are_explicit(self):
        for text in ('on', 'true', 'TRUE', '1'):
            self.assertTrue(self.resolve(env={**self.env, 'HF_UPLOAD_ENABLED': text})['enabled'])
        for text in ('off', 'false', 'FALSE', '0'):
            self.assertFalse(self.resolve(env={'HF_UPLOAD_ENABLED': text})['enabled'])
        for text in ('', 'yes', ' true', '2'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.resolve(env={**self.env, 'HF_UPLOAD_ENABLED': text})

    def test_pilot_private_override_is_explicit_and_supported_in_literal_env(self):
        envpath = self.root / '.env'
        envpath.write_text('HF_PRIVATE_ALL=true\n')
        env = dict(self.env)
        env_config.load_env_file(envpath, environ=env)
        self.assertTrue(self.resolve(env=env)['private_all'])
        for value in ('on', 'TRUE', '1'):
            self.assertTrue(self.resolve(env={**self.env, 'HF_PRIVATE_ALL': value})['private_all'])
        for value in ('off', 'FALSE', '0'):
            self.assertFalse(self.resolve(env={**self.env, 'HF_PRIVATE_ALL': value})['private_all'])
        for value in ('', 'yes', ' true', '2'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'HF_PRIVATE_ALL'):
                self.resolve(env={**self.env, 'HF_PRIVATE_ALL': value})

    def test_hub_names_exclude_path_traversal_and_reserved_forms(self):
        for name in ('../repo', 'a/b', '.hidden', 'trailing.', 'leading-', '-leading',
                     'bad..name', 'bad--name', 'repo.git', 'name space', 'x' * 97):
            with self.subTest(name=name), self.assertRaises(ValueError):
                hf_config.validate_slug(name, 'name')
        for name in ('x', 'my-team', 'Qwen3_4B', 'v1.2-model'):
            self.assertEqual(hf_config.validate_slug(name, 'name'), name)

    def test_upload_settings_not_forwarded_to_gpu_driver(self):
        settings = run_settings.load_settings(self.path)
        args = run_settings.settings_arguments('train', [], settings, environ={})
        self.assertFalse(any(value.startswith(('--huggingface-', '--hf-')) for value in args))

    def test_inputs_are_not_mutated_and_mapping_settings_are_supported(self):
        settings = run_settings.load_settings(self.path).values
        before = copy.deepcopy(settings)
        overrides = dict(hf_epochs='1,4', hf_upload='on')
        env = dict(self.env)
        result = hf_config.resolve_hf_config('train', settings, overrides, env)
        result['epochs'].append(5)
        self.assertEqual(settings, before)
        self.assertEqual(env, self.env)
        self.assertEqual(overrides, dict(hf_epochs='1,4', hf_upload='on'))

    def test_unknown_stage_rejected(self):
        with self.assertRaises(ValueError):
            self.resolve(stage='upload')


if __name__ == '__main__':
    unittest.main()
