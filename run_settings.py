"""Strict, secret-free TOML defaults for the portable launcher (Python 3.11+)."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from string import Formatter
import tomllib

from hf_config import HF_DEFAULTS, validate_hf_setting

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    'training': dict(model='qwen4b', arm='local_positive', finetune_mode='full',
                     model_path='models/{model}', data='data/train.json',
                     out='runs/{model}/{finetune_mode}/comparison_seed{seed}/{arm}',
                     batch_size=64, rollouts=8, epochs=4, seed=42, alpha=1.,
                     learning_rate='auto', warmup_steps=0, kl_coefficient=0., sampler_max_num_seqs=32, sampler_eager=False,
                     generation_timeout=3600, smoke_cap=128),
    'validation': dict(mode='slurm', data='data/val.json', every_epochs=1, rollouts=8,
                       partition='', account='', qos=''),
    'wandb': dict(mode='online', project='stride-iclr2027', group='stride-comparison', entity=''),
    'huggingface': {key: list(value) if isinstance(value, list) else value for key, value in HF_DEFAULTS.items()},
}
CHOICES = {
    ('training', 'model'): ('qwen4b', 'qwen17b'),
    ('training', 'arm'): ('local_positive', 'single_step', 'token_uniform', 'grpo'),
    ('training', 'finetune_mode'): ('full', 'lora'),
    ('training', 'sampler_max_num_seqs'): (16, 32, 64),
    ('validation', 'mode'): ('slurm', 'off'),
    ('wandb', 'mode'): ('online', 'offline', 'disabled'),
}
TRAIN_ENV = {
    'model': 'MODEL', 'arm': 'ARM', 'finetune_mode': 'FINETUNE_MODE', 'seed': 'SEED',
    'batch_size': 'BATCH_SIZE', 'rollouts': 'ROLLOUTS', 'epochs': 'EPOCHS',
    'alpha': 'ALPHA', 'learning_rate': 'LEARNING_RATE',
    'warmup_steps': 'WARMUP_STEPS', 'kl_coefficient': 'KL_COEFFICIENT',
}
RUNTIME_ENV = {
    'validation': {key: 'VALIDATION_' + key.upper() for key in DEFAULTS['validation']},
    'wandb': dict(mode='WANDB_MODE', project='WANDB_PROJECT', group='WANDB_RUN_GROUP', entity='WANDB_ENTITY'),
}
PATH_FIELDS = {('training', 'model_path'), ('training', 'data'), ('training', 'out'), ('validation', 'data')}


@dataclass(frozen=True)
class RunSettings:
    path: Path
    values: dict


def _option(arguments, key):
    values = []
    for i, item in enumerate(arguments):
        if item == key:
            if i + 1 == len(arguments) or arguments[i + 1].startswith('--'):
                raise ValueError(f'Missing value for {key}')
            values.append(arguments[i + 1])
        elif item.startswith(key + '='):
            values.append(item.split('=', 1)[1])
    if len(values) > 1:
        raise ValueError(f'Duplicate option {key}')
    return values[0] if values else None


def _check(section, name, value, *, check_template=True):
    label = f'{section}.{name}'
    expected = type(DEFAULTS[section][name])
    if name == 'learning_rate' and section == 'training':
        valid = value == 'auto' if type(value) is str else type(value) in (int, float) and math.isfinite(value) and value > 0
    elif expected is float:
        valid = type(value) in (float, int) and math.isfinite(value)
    else:
        valid = type(value) is expected
    if not valid:
        raise ValueError(f'Invalid type or value for {label}')
    if isinstance(value, str) and any(ord(char) < 32 for char in value):
        raise ValueError(f'{label} must be a single-line value')
    if (section, name) in CHOICES and value not in CHOICES[section, name]:
        raise ValueError(f'Invalid choice for {label}')
    if section == 'training':
        if name == 'batch_size' and (value < 1 or value % 4):
            raise ValueError('training.batch_size must be positive and divisible by four')
        if name == 'seed' and not 0 <= value < 2**32:
            raise ValueError('training.seed must be in [0, 2**32)')
        if name == 'rollouts' and not 2 <= value <= 64:
            raise ValueError('training.rollouts must be in [2, 64]')
        if name in ('epochs', 'generation_timeout') and value < 1:
            raise ValueError(f'{label} must be positive')
        if name in ('alpha', 'kl_coefficient', 'warmup_steps') and value < 0:
            raise ValueError(f'{label} must be nonnegative')
        if name == 'smoke_cap' and not 1 <= value <= 8192:
            raise ValueError('training.smoke_cap must be in [1, 8192]')
    if section == 'validation':
        if name == 'every_epochs' and value < 1:
            raise ValueError('validation.every_epochs must be positive')
        if name == 'rollouts' and not 1 <= value <= 64:
            raise ValueError('validation.rollouts must be in [1, 64]')
    if (section, name) in PATH_FIELDS and not value.strip():
        raise ValueError(f'{label} must be nonempty')
    if check_template and (section, name) in PATH_FIELDS:
        _path_template(value)
    if section == 'wandb' and name == 'project' and not value.strip():
        raise ValueError('wandb.project must be nonempty')
    if section == 'huggingface':
        validate_hf_setting(name, value)


def load_settings(path=None):
    filename = (ROOT / 'run_settings.toml' if path is None else Path(path).expanduser())
    if filename.is_symlink() or not filename.is_file():
        raise ValueError('Run settings must be an existing regular TOML file')
    try:
        with filename.open('rb') as stream:
            payload = tomllib.load(stream)
    except (OSError, ValueError, UnicodeError):
        # Do not include the parser's excerpt: a user may have pasted a secret.
        raise ValueError('Cannot read run settings: invalid TOML or encoding') from None
    result = {section: {name: list(value) if isinstance(value, list) else value
                        for name, value in values.items()} for section, values in DEFAULTS.items()}
    for section, values in payload.items():
        if section not in DEFAULTS or not isinstance(values, dict):
            raise ValueError('Unknown or invalid run settings section')
        for name, value in values.items():
            if name not in DEFAULTS[section]:
                raise ValueError(f'Unknown setting in [{section}]')
            _check(section, name, value)
            result[section][name] = value
    return RunSettings(filename.resolve(), result)


def _external_value(section, name, value):
    """Parse known string CLI/environment values without permissive whitespace."""
    if not isinstance(value, str):
        raise ValueError(f'Invalid external setting for {section}.{name}')
    kind = type(DEFAULTS[section][name])
    try:
        if kind is int:
            if not re.fullmatch(r'[0-9]+', value):
                raise ValueError
            result = int(value)
        elif kind is float or (name == 'learning_rate' and value != 'auto'):
            if value != value.strip():
                raise ValueError
            result = float(value)
        else:
            result = value
    except (ValueError, OverflowError):
        raise ValueError(f'Invalid external setting for {section}.{name}') from None
    _check(section, name, result, check_template=False)
    return result


def _path_template(value):
    try:
        for _, field, spec, conversion in Formatter().parse(value):
            if field is not None and (field not in ('model', 'arm', 'finetune_mode', 'seed') or spec or conversion):
                raise ValueError
    except (ValueError, KeyError, IndexError):
        raise ValueError('Settings paths allow only {model}, {arm}, {finetune_mode}, and {seed}') from None


def _path(value, settings, effective):
    _path_template(value)
    expanded = value.format_map(effective)
    path = Path(expanded).expanduser()
    return str((settings.path.parent / path).resolve())


def settings_arguments(stage, arguments, settings, *, environ=None):
    """Apply CLI > exported env > TOML defaults, leaving runtime env to its loader.

    Call env_config.runtime_arguments on the returned list next, so that .env
    W&B and validation settings still override TOML. Eval inherits W&B only.
    """
    if stage not in ('train', 'smoke', 'eval'):
        raise ValueError('Unknown launcher stage')
    env = os.environ if environ is None else environ
    result = list(arguments)
    effective = {}
    if stage != 'eval':
        for name, value in settings.values['training'].items():
            flag = '--' + name.replace('_', '-')
            if name == 'sampler_eager':
                positive = result.count('--sampler-eager')
                negative = result.count('--no-sampler-eager')
                if positive + negative > 1 or any(x.startswith(('--sampler-eager=', '--no-sampler-eager=')) for x in result):
                    raise ValueError('Use one of --sampler-eager or --no-sampler-eager')
                if negative:
                    result.remove('--no-sampler-eager')
                elif not positive and value:
                    result.append(flag)
                continue
            explicit = _option(result, flag)
            key = TRAIN_ENV.get(name)
            external = explicit if explicit is not None else env.get(key) if key else None
            actual = _external_value('training', name, external) if external is not None else value
            if name in ('model', 'arm', 'finetune_mode', 'seed'):
                effective[name] = actual
        for name, value in settings.values['training'].items():
            if name == 'sampler_eager':
                continue
            flag = '--' + name.replace('_', '-')
            explicit = _option(result, flag)
            if name == 'learning_rate' and explicit == 'auto':
                for index, item in enumerate(result):
                    if item == flag:
                        del result[index:index + 2]
                        break
                    if item.startswith(flag + '='):
                        del result[index]
                        break
                # The explicit auto override deliberately ignores a numeric TOML LR.
                continue
            if explicit is not None:
                continue
            key = TRAIN_ENV.get(name)
            value = _external_value('training', name, env[key]) if key is not None and key in env else value
            if name == 'learning_rate' and value == 'auto':
                continue
            if ('training', name) in PATH_FIELDS:
                value = _path(value, settings, effective)
                if stage == 'smoke' and name == 'out':
                    value = str(Path(value).with_name(Path(value).name + '_smoke'))
            result.extend((flag, str(value)))
    sections = ('wandb',) if stage == 'eval' else ('validation', 'wandb')
    for section in sections:
        for name, value in settings.values[section].items():
            flag = '--' + section + '-' + name.replace('_', '-')
            # Actual CLI and runtime environment always beat file defaults.
            if _option(result, flag) is not None or RUNTIME_ENV[section][name] in env:
                continue
            if stage == 'smoke' and name == 'mode':
                value = 'off' if section == 'validation' else 'disabled'
            if (section, name) in PATH_FIELDS:
                value = _path(value, settings, effective)
            if value != '':
                result.extend((flag, str(value)))
    return result
