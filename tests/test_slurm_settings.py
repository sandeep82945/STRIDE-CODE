"""Execute real shell wrappers and resolve root settings without GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SlurmSettingsTests(unittest.TestCase):
    def invoke(self, script='slurm_run.sbatch', *, arguments=(), **settings):
        with tempfile.TemporaryDirectory(prefix='stride shell ') as folder:
            root = Path(folder)
            (root / '.venv/bin').mkdir(parents=True)
            (root / 'scripts').mkdir()
            shutil.copy(ROOT/'scripts/slurm_run.sbatch', root/'scripts/slurm_run.sbatch')
            shutil.copy(ROOT/'run_settings.toml', root/'run_settings.toml')
            # Python shebang points to the actual running environment, no GPU imports.
            import sys
            recorder = root/'.venv/bin/python'
            recorder.write_text(f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
from run_settings import load_settings, settings_arguments
argv = sys.argv[1:]
try:
    if len(argv) >= 2 and argv[1] in ('train', 'smoke'):
        argv = argv[:2] + settings_arguments(argv[1], argv[2:], load_settings(Path.cwd()/'run_settings.toml'), environ=os.environ)
except ValueError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(2)
# Compare paths independently of the temporary fixture directory.
prefix = str(Path.cwd()) + '/'
argv = [value[len(prefix):] if value.startswith(prefix) else value for value in argv]
print(json.dumps(argv))
""")
            recorder.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in ('ARM', 'MODEL', 'FINETUNE_MODE', 'SEED', 'STAGE', 'SLURM_ARRAY_TASK_ID')}
            env.update(SLURM_SUBMIT_DIR=folder, **settings)
            return subprocess.run(['bash', str(ROOT/script if script == 'run.sbatch' else ROOT/'scripts'/script), *arguments], env=env, capture_output=True, text=True)

    def test_root_entrypoint_uses_same_resolved_settings(self):
        for mode in ('full', 'lora'):
            root_run = self.invoke('run.sbatch', FINETUNE_MODE=mode, ARM='token_uniform')
            legacy_run = self.invoke(FINETUNE_MODE=mode, ARM='token_uniform')
            self.assertEqual(root_run.returncode, 0, root_run.stderr)
            self.assertEqual(json.loads(root_run.stdout), json.loads(legacy_run.stdout))
            self.assertEqual(json.loads(root_run.stdout)[json.loads(root_run.stdout).index('--finetune-mode')+1], mode)

    def test_cli_overrides_root_defaults_and_array_environment(self):
        result = self.invoke('slurm_compare.sbatch', SLURM_ARRAY_TASK_ID='0',
                             FINETUNE_MODE='full', arguments=('--finetune-mode', 'lora', '--batch-size', '32'))
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertEqual(args[args.index('--finetune-mode')+1], 'lora')
        self.assertEqual(args[args.index('--batch-size')+1], '32')
        self.assertEqual(args.count('--finetune-mode'), 1)
        self.assertEqual(args.count('--batch-size'), 1)

    def test_changing_arm_is_the_only_difference(self):
        reference = None
        for arm in ('local_positive', 'single_step', 'token_uniform', 'grpo'):
            run = self.invoke(ARM=arm)
            self.assertEqual(run.returncode, 0, run.stderr)
            args = json.loads(run.stdout)
            self.assertEqual(args[args.index('--arm')+1], arm)
            self.assertNotIn('--adapter', args)
            self.assertEqual(args[args.index('--warmup-steps') + 1], '0')
            self.assertEqual(args[args.index('--kl-coefficient') + 1], '0.0')
            self.assertNotIn('--format-coefficient', args)
            self.assertEqual(args[args.index('--out')+1], f'runs/qwen4b/full/comparison_seed42/{arm}')
            args[args.index('--arm')+1] = 'ARM'
            args[args.index('--out')+1] = 'OUT'
            if reference is None: reference = args
            else: self.assertEqual(args, reference)

    def test_second_model_starts_directly_without_adapter(self):
        run = self.invoke(ARM='single_step', MODEL='qwen17b')
        self.assertEqual(run.returncode, 0, run.stderr)
        args = json.loads(run.stdout)
        self.assertEqual(args[1], 'train')
        self.assertEqual(args[args.index('--arm')+1], 'single_step')
        self.assertNotIn('--adapter', args)
        self.assertEqual(args[args.index('--model-path')+1], 'models/qwen17b')
        self.assertEqual(args[args.index('--out')+1], 'runs/qwen17b/full/comparison_seed42/single_step')

    def test_array_selects_exactly_four_settings(self):
        for index, arm in enumerate(('local_positive', 'single_step', 'token_uniform', 'grpo')):
            run = self.invoke('slurm_compare.sbatch', SLURM_ARRAY_TASK_ID=str(index))
            self.assertEqual(run.returncode, 0, run.stderr)
            args = json.loads(run.stdout)
            self.assertEqual(args[args.index('--arm')+1], arm)

    def test_nondefault_seed_is_shared_by_every_arm(self):
        reference = None
        for arm in ('local_positive', 'single_step', 'token_uniform', 'grpo'):
            result = self.invoke(ARM=arm, SEED='43')
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(result.stdout)
            self.assertEqual(args[args.index('--seed')+1], '43')
            self.assertEqual(args[args.index('--data')+1], 'data/train.json')
            self.assertEqual(args[args.index('--out')+1], f'runs/qwen4b/full/comparison_seed43/{arm}')
            args[args.index('--arm')+1] = 'ARM'
            args[args.index('--out')+1] = 'OUT'
            if reference is None: reference = args
            else: self.assertEqual(args, reference)

    def test_array_nondefault_seed_and_frozen_data_are_shared(self):
        for index, arm in enumerate(('local_positive', 'single_step', 'token_uniform', 'grpo')):
            result = self.invoke('slurm_compare.sbatch', SLURM_ARRAY_TASK_ID=str(index),
                                 SEED='43', MODEL='qwen17b', ARM='ignored-by-array')
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(result.stdout)
            self.assertEqual(args[args.index('--arm')+1], arm)
            self.assertEqual(args[args.index('--seed')+1], '43')
            self.assertEqual(args[args.index('--data')+1], 'data/train.json')
            self.assertEqual(args[args.index('--model-path')+1], 'models/qwen17b')
            self.assertEqual(args[args.index('--out')+1], f'runs/qwen17b/full/comparison_seed43/{arm}')

    def test_evaluation_inherits_seed_and_explicit_override_is_not_duplicated(self):
        for supplied, expected in (((), '43'), (('--seed', '44'), '44'), (('--seed=44',), '44')):
            with self.subTest(supplied=supplied):
                result = self.invoke('slurm_eval.sbatch', SEED='43', arguments=supplied)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(result.stdout)
                self.assertEqual(args[:2], ['run.py', 'eval'])
                seeds = []
                for index, arg in enumerate(args):
                    if arg == '--seed': seeds.append(args[index+1])
                    elif arg.startswith('--seed='): seeds.append(arg.split('=', 1)[1])
                self.assertEqual(seeds, [expected])

    def test_invalid_settings_fail_without_launch(self):
        for settings in (dict(FINETUNE_MODE='bad'), dict(ARM='centered'), dict(MODEL='unknown'), dict(STAGE='eval'), dict(STAGE='warmup'),
                         dict(SEED='bad'), dict(SEED='-1'), dict(SEED='4.3'),
                         dict(SEED='43; exit 0'), dict(SEED=' 43')):
            self.assertEqual(self.invoke(**settings).returncode, 2)
        self.assertEqual(self.invoke('slurm_compare.sbatch', SLURM_ARRAY_TASK_ID='4').returncode, 2)


if __name__ == '__main__':
    unittest.main()
