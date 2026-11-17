"""Literal local configuration; secrets stay in the child environment only."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parent
KEYS = {'WANDB_API_KEY', 'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_RUN_GROUP',
        'WANDB_MODE', 'VALIDATION_MODE', 'VALIDATION_DATA',
        'VALIDATION_EVERY_EPOCHS', 'VALIDATION_ROLLOUTS',
        'VALIDATION_PARTITION', 'VALIDATION_ACCOUNT', 'VALIDATION_QOS',
        'HF_TOKEN', 'HF_USERNAME', 'HF_UPLOAD_ENABLED', 'HF_PRIVATE_ALL'}


def load_env_file(path=None, *, environ=None):
    """Load an optional .env without shell execution; exported values win.

    Only documented keys are allowed. Parse the entire file before changing the
    environment, and never include input values in a configuration error.
    """
    target = os.environ if environ is None else environ
    filename = ROOT / '.env' if path is None else Path(path).expanduser()
    if not filename.exists() and not filename.is_symlink():
        if path is not None:
            raise ValueError('The specified --env-file does not exist')
        return {}
    if filename.is_symlink() or not filename.is_file():
        raise ValueError('The .env configuration must be a regular file')
    try:
        contents = filename.read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        raise ValueError('Unable to read the .env configuration') from None
    values = {}
    for number, raw in enumerate(contents.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        key, separator, value = line.partition('=')
        key = key.strip()
        if not separator or key not in KEYS or key in values:
            raise ValueError(f'Unknown, duplicate or malformed .env key on line {number}')
        try:
            fields = shlex.split(value, comments=True, posix=True)
        except ValueError:
            raise ValueError(f'Invalid .env quoting on line {number}') from None
        if len(fields) > 1:
            raise ValueError(f'Quote values containing spaces on .env line {number}')
        values[key] = fields[0] if fields else ''
    for key, value in values.items():
        target.setdefault(key, value)
    return values


def _option(arguments, key):
    found = []
    for index, item in enumerate(arguments):
        if item == key:
            if index + 1 == len(arguments) or arguments[index + 1].startswith('--'):
                raise ValueError(f'Missing value for {key}')
            found.append(arguments[index + 1])
        elif item.startswith(key + '='):
            found.append(item.split('=', 1)[1])
    if len(found) > 1:
        raise ValueError(f'Duplicate option {key}')
    return found[0] if found else None


def runtime_arguments(stage, arguments, *, environ=None, require_key=True):
    """Resolve CLI > exported environment > .env > defaults, without credentials.

    The supplied environment receives the resolved W&B mode so the trainer and
    its validation jobs see the same configuration. No API key is ever returned
    in the argument list. Dry runs may omit a key, but cannot print one.
    """
    env = os.environ if environ is None else environ
    result = list(arguments)
    token = env.get('WANDB_API_KEY', '')
    secrets = [env[key] for key in ('WANDB_API_KEY', 'HF_TOKEN') if env.get(key)]
    if any(secret in argument for secret in secrets for argument in result):
        raise ValueError('Credentials must not appear in command-line arguments or metadata')
    if any(argument.split('=', 1)[0] in ('--wandb-api-key', '--api-key', '--hf-token') for argument in result):
        raise ValueError('Put credentials in .env or the environment, never the command line')

    def setting(flag, key, default='', *, optional=False):
        explicit = _option(result, flag)
        value = explicit if explicit is not None else env.get(key, default)
        if any(secret in value for secret in secrets):
            raise ValueError('Credentials must not appear in nonsecret configuration')
        if any(ord(char) < 32 for char in value):
            raise ValueError(f'{flag} must be a single-line value')
        if explicit is None and (value or not optional):
            result.extend((flag, value))
        return value

    mode = setting('--wandb-mode', 'WANDB_MODE', 'disabled' if stage == 'smoke' else 'online')
    project = setting('--wandb-project', 'WANDB_PROJECT', 'stride-iclr2027')
    setting('--wandb-group', 'WANDB_RUN_GROUP', 'stride-comparison')
    setting('--wandb-entity', 'WANDB_ENTITY', optional=True)
    if mode not in ('online', 'offline', 'disabled'):
        raise ValueError('--wandb-mode / WANDB_MODE must be online, offline, or disabled')
    if mode != 'disabled' and not project.strip():
        raise ValueError('Set a nonempty WANDB_PROJECT or --wandb-project')
    if mode == 'online' and require_key:
        if (len(token) < 8 or any(char.isspace() or ord(char) < 32 for char in token)
                or token.lower().startswith(('your_', 'paste_', 'replace_'))):
            raise ValueError('Set WANDB_API_KEY in .env or the environment before launching online W&B')
    env['WANDB_MODE'] = mode
    # Never let a shell export enable automatic source uploads (which could
    # inadvertently include local configuration). Metrics logging still works.
    env['WANDB_DISABLE_CODE'] = 'true'

    if stage != 'eval':
        validation = setting('--validation-mode', 'VALIDATION_MODE', 'off' if stage == 'smoke' else 'slurm')
        if validation not in ('slurm', 'off'):
            raise ValueError('--validation-mode / VALIDATION_MODE must be slurm or off')
        validation_path = setting('--validation-data', 'VALIDATION_DATA', str(ROOT / 'data' / 'val.json'))
        if validation == 'slurm' and not validation_path:
            raise ValueError('Set VALIDATION_DATA or --validation-data for validation')
        for flag, key, default, maximum in (
                ('--validation-every-epochs', 'VALIDATION_EVERY_EPOCHS', '1', None),
                ('--validation-rollouts', 'VALIDATION_ROLLOUTS', '8', 64)):
            value = setting(flag, key, default)
            if not re.fullmatch(r'[0-9]+', value) or int(value) < 1 or (maximum and int(value) > maximum):
                raise ValueError(f'{flag} must be a positive integer' + (f' no greater than {maximum}' if maximum else ''))
        for suffix in ('partition', 'account', 'qos'):
            setting('--validation-' + suffix, 'VALIDATION_' + suffix.upper(), optional=True)
    return result
