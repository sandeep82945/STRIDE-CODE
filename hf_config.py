"""Secret-free Hugging Face upload configuration for the portable launcher.

This module does not import the Hub client or perform network operations. Tokens
remain in the child environment and are never part of the returned upload plan.
"""
from __future__ import annotations

import json
import os
import re


HF_DEFAULTS = dict(enabled=True, epochs=[2, 3, 4],
                   repo_prefix='stride-iclr2027', description='', retries=3)
_OVERRIDES = dict(enabled='hf_upload', epochs='hf_epochs',
                  repo_prefix='hf_repo_prefix', description='hf_description',
                  retries='hf_retries')
_TOKEN_KEYS = ('HF_TOKEN', 'WANDB_API_KEY')


def validate_slug(value, label):
    """Validate one Hub repository-ID component without echoing its value."""
    if (type(value) is not str or not 1 <= len(value) <= 96
            or not re.fullmatch(r'[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?', value)
            or '..' in value or '--' in value or value.endswith('.git')):
        raise ValueError(f'{label} must be a valid Hugging Face name (1–96 characters)')
    return value


def validate_hf_setting(name, value):
    """Strict TOML and resolved-value checks, shared by both config loaders."""
    label = f'huggingface.{name}'
    if name not in HF_DEFAULTS or type(value) is not type(HF_DEFAULTS[name]):
        raise ValueError(f'Invalid type or value for {label}')
    if name == 'epochs':
        if (not value or any(type(epoch) is not int or epoch < 1 for epoch in value)
                or value != sorted(set(value))):
            raise ValueError('huggingface.epochs must contain increasing, unique positive integers')
    elif name == 'repo_prefix':
        validate_slug(value, label)
    elif name == 'description':
        if len(value) > 4000 or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError('huggingface.description must be one line of at most 4000 characters')
    elif name == 'retries' and not 1 <= value <= 10:
        raise ValueError('huggingface.retries must be in [1, 10]')


def _parse_override(name, value):
    if type(value) is not str:
        raise ValueError(f'Invalid external setting for huggingface.{name}')
    if name == 'enabled':
        if value not in ('on', 'off'):
            raise ValueError('--hf-upload must be on or off')
        result = value == 'on'
    elif name == 'epochs':
        if not re.fullmatch(r'[0-9]+(?:,[0-9]+)*', value):
            raise ValueError('--hf-epochs must be comma-separated positive integers')
        result = [int(item) for item in value.split(',')]
    elif name == 'retries':
        if not re.fullmatch(r'[0-9]+', value):
            raise ValueError('--hf-retries must be a positive integer')
        result = int(value)
    else:
        result = value
    validate_hf_setting(name, result)
    return result


def _env_enabled(value):
    if type(value) is not str or value.lower() not in ('true', 'false', 'on', 'off', '1', '0'):
        raise ValueError('HF_UPLOAD_ENABLED must be true or false (also on/off or 1/0)')
    return value.lower() in ('true', 'on', '1')


def resolve_hf_config(stage, settings, overrides=None, environ=None, *, require_credentials=True):
    """Resolve CLI > exported env > .env > TOML into a nonsecret upload plan.

    ``settings`` is a RunSettings instance, its values dict, or None for
    ``--no-settings``. ``overrides`` is a dict such as ``vars(parsed_args)``.
    The caller must already have merged .env with exported environment values.
    Eval/smoke never upload. Without TOML, upload is opt-in. Dry runs can omit
    credentials by setting require_credentials=False and never access the Hub.
    """
    if stage not in ('train', 'smoke', 'eval'):
        raise ValueError('Unknown launcher stage')
    env = os.environ if environ is None else environ
    values = {key: list(value) if isinstance(value, list) else value
              for key, value in HF_DEFAULTS.items()}
    if settings is None:
        values['enabled'] = False
    else:
        configured = settings.values if hasattr(settings, 'values') and not isinstance(settings, dict) else settings
        if not isinstance(configured, dict) or not isinstance(configured.get('huggingface', {}), dict):
            raise ValueError('Invalid Hugging Face settings')
        for name, value in configured.get('huggingface', {}).items():
            validate_hf_setting(name, value)
            values[name] = list(value) if isinstance(value, list) else value
    if 'HF_UPLOAD_ENABLED' in env and stage == 'train':
        values['enabled'] = _env_enabled(env['HF_UPLOAD_ENABLED'])
    cli = {} if overrides is None else overrides
    if not isinstance(cli, dict):
        raise ValueError('Hugging Face overrides must be a mapping')
    for name, flag in _OVERRIDES.items():
        if cli.get(flag) is not None:
            values[name] = _parse_override(name, cli[flag])
    for name, value in values.items():
        validate_hf_setting(name, value)
    private_all = env.get('HF_PRIVATE_ALL', 'false')
    if type(private_all) is not str or private_all.lower() not in ('true', 'false', 'on', 'off', '1', '0'):
        raise ValueError('HF_PRIVATE_ALL must be true or false (also on/off or 1/0)')
    values['private_all'] = private_all.lower() in ('true', 'on', '1')
    values['enabled'] = bool(values['enabled'] and stage == 'train')
    values['namespace'] = env.get('HF_USERNAME', '') if values['enabled'] else ''
    serialized = json.dumps(values, ensure_ascii=False)
    for key in _TOKEN_KEYS:
        secret = env.get(key, '')
        if secret and secret in serialized:
            raise ValueError('Credentials must not appear in Hugging Face upload settings')
    if values['enabled']:
        if values['namespace']:
            validate_slug(values['namespace'], 'HF_USERNAME')
        elif require_credentials:
            raise ValueError('Set HF_USERNAME in .env or the environment before enabling Hugging Face upload')
        token = env.get('HF_TOKEN', '')
        if require_credentials and (len(token) < 8
                or any(char.isspace() or ord(char) < 32 for char in token)
                or token.lower().startswith(('your_', 'paste_', 'replace_'))):
            raise ValueError('Set HF_TOKEN in .env or the environment before enabling Hugging Face upload')
    return values
