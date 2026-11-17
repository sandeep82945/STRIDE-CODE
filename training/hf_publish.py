"""Publish verified, completed epoch exports to Hugging Face (CPU only).

Each epoch is an independently loadable model/adapter repository.  This worker
is launched after training in its own process, never from a learner rank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid


HUB_ENDPOINT = 'https://huggingface.co'
PRIVATE_ARMS = frozenset(('local_positive', 'single_step', 'token_uniform'))
EXPORT_KEYS = frozenset(('model', 'arm', 'finetune_mode', 'seed', 'K', 'alpha',
                         'epoch', 'step', 'training_questions', 'batch_size',
                         'total_epochs', 'dataset_sha256', 'run_fingerprint',
                         'base_model', 'base_revision', 'files'))
TOKENIZER_FILES = frozenset(('tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
    'tokenizer.model', 'chat_template.jinja', 'chat_template.jinja2'))


class PublishError(ValueError):
    """An actionable diagnostic that does not include a service response."""


def private_all(environ=None):
    """Opt-in pilot privacy never makes a STRIDE checkpoint public."""
    env = os.environ if environ is None else environ
    value = env.get('HF_PRIVATE_ALL', 'false')
    if type(value) is not str or value.lower() not in ('true', 'false', 'on', 'off', '1', '0'):
        raise PublishError('HF_PRIVATE_ALL must be true or false (also on/off or 1/0)')
    return value.lower() in ('true', 'on', '1')


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _secret_guard(text):
    for name in ('HF_TOKEN', 'WANDB_API_KEY'):
        value = os.environ.get(name, '').strip()
        if value and value in text:
            raise PublishError('Credentials must not appear in upload names, descriptions or metadata')


def _safe_error(exc):
    if isinstance(exc, PublishError):
        message = str(exc)
    else:
        code = _field(getattr(exc, 'response', None), 'status_code')
        message = f'Hugging Face operation failed ({type(exc).__name__}'
        message += f', HTTP {code})' if isinstance(code, int) else ')'
        message += '; check credentials, namespace access, connectivity and retry'
    for name in ('HF_TOKEN', 'WANDB_API_KEY'):
        secret = os.environ.get(name, '').strip()
        if secret:
            message = message.replace(secret, '[redacted]')
    return message


def _namespace(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', value):
        raise PublishError('HF namespace must be a user or organization name without a slash')
    if '..' in value or '--' in value or value.endswith(('.', '-')):
        raise PublishError('Invalid Hugging Face namespace')
    _secret_guard(value)
    return value


def _token(value=None):
    key = (value if value is not None else os.environ.get('HF_TOKEN', '')).strip()
    if not key:
        raise PublishError('Set HF_TOKEN in .env or the environment to enable automatic upload')
    return key


def _api(api, token):
    if api is None:
        from huggingface_hub import HfApi
        return HfApi(endpoint=HUB_ENDPOINT, token=token)
    endpoint = getattr(api, 'endpoint', HUB_ENDPOINT)
    if endpoint.rstrip('/') != HUB_ENDPOINT:
        raise PublishError('Checkpoint uploads require the official Hugging Face endpoint')
    return api


def preflight_auth(namespace, token=None, api=None):
    """Check identity and namespace membership without creating a repository.

    Fine-grained token permissions are finally checked by repository creation;
    whoami cannot guarantee permission to create a future repository.
    """
    namespace, token = _namespace(namespace), _token(token)
    api = _api(api, token)
    try:
        identity = api.whoami(token=token)
    except Exception as exc:
        raise PublishError(_safe_error(exc)) from None
    name = identity.get('name')
    organizations = {item.get('name') for item in identity.get('orgs', []) if isinstance(item, dict)}
    if namespace != name and namespace not in organizations:
        raise PublishError('HF_USERNAME must match the token owner or one of their organizations')
    role = identity.get('auth', {}).get('accessToken', {}).get('role')
    if role == 'read':
        raise PublishError('HF_TOKEN is read-only; use a token with repository write/create access')
    return {'namespace': namespace, 'authenticated_user': name, 'token_role': role}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _git_blob(path):
    path = Path(path)
    digest = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _allowed_file(name, mode):
    if name in TOKENIZER_FILES or name in ('README.md', 'export_manifest.json', 'training_metadata.json'):
        return True
    if mode == 'lora':
        return name in ('adapter_config.json', 'adapter_model.safetensors')
    return name in ('config.json', 'generation_config.json', 'model.safetensors',
                    'model.safetensors.index.json') or bool(re.fullmatch(r'model-\d+-of-\d+\.safetensors', name))


def verify_export(item, staging):
    """Recheck the exact allowlist and hashes immediately before networking."""
    if set(item) != EXPORT_KEYS | {'folder'}:
        raise PublishError('Checkpoint export metadata has unexpected or missing fields')
    manifest = {key: value for key, value in item.items() if key != 'folder'}
    _secret_guard(json.dumps(manifest, sort_keys=True))
    mode, arm = item['finetune_mode'], item['arm']
    if mode not in ('full', 'lora') or arm not in PRIVATE_ARMS | {'grpo'}:
        raise PublishError('Checkpoint export has an unsupported method or finetuning mode')
    for name in ('epoch', 'step', 'K', 'training_questions', 'batch_size', 'total_epochs'):
        if type(item[name]) is not int or item[name] < 1:
            raise PublishError('Checkpoint export has an invalid numeric setting')
    if type(item['seed']) is not int or item['seed'] < 0:
        raise PublishError('Checkpoint export has an invalid seed')
    if not isinstance(item['alpha'], (int, float)) or not math.isfinite(item['alpha']) or item['alpha'] < 0:
        raise PublishError('Checkpoint export has an invalid diversity coefficient')
    if any(not isinstance(item[key], str) or not re.fullmatch('[0-9a-f]{64}', item[key])
           for key in ('dataset_sha256', 'run_fingerprint')):
        raise PublishError('Checkpoint export must contain dataset and run fingerprints')
    folder = Path(item['folder'])
    staging = Path(staging).resolve()
    if folder.is_symlink() or not folder.is_dir() or not folder.resolve().is_relative_to(staging):
        raise PublishError('Checkpoint export folder must be inside the run export staging directory')
    # No nested directories, hidden files, symbolic links or unexpected artifacts.
    paths = list(folder.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise PublishError('Checkpoint export contains a symbolic link or unexpected directory')
    expected = item['files']
    if not isinstance(expected, dict) or not expected:
        raise PublishError('Checkpoint export has no file manifest')
    if set(path.name for path in paths) != set(expected) | {'export_manifest.json'}:
        raise PublishError('Checkpoint export contains unlisted or missing files')
    hashes = {}
    for name in sorted(set(expected) | {'export_manifest.json'}):
        if not _allowed_file(name, mode):
            raise PublishError('Checkpoint export contains a file outside the model/tokenizer allowlist')
        path = folder / name
        hashes[name] = _sha(path)
        if name in expected and hashes[name] != expected[name]:
            raise PublishError('Checkpoint export changed after it was prepared; refusing to upload')
        if name.endswith(('.json', '.md', '.jinja', '.jinja2', '.txt')):
            _secret_guard(path.read_text())
    try:
        actual = json.loads((folder / 'export_manifest.json').read_text())
    except (ValueError, UnicodeError):
        raise PublishError('Checkpoint export manifest is not valid JSON') from None
    if actual != manifest:
        raise PublishError('Checkpoint export manifest does not match the prepared payload')
    required = {'README.md', 'tokenizer_config.json'}
    required |= {'adapter_config.json', 'adapter_model.safetensors'} if mode == 'lora' else {'config.json'}
    if not required.issubset(expected) or (mode == 'full' and not any(name.endswith('.safetensors') for name in expected)):
        raise PublishError('Checkpoint export is missing required model/adapter or tokenizer files')
    return hashes


def repository_id(namespace, prefix, item):
    """Readable setting names plus a collision-resistant run/epoch suffix."""
    _namespace(namespace)
    _secret_guard(str(prefix))
    def slug(value):
        return re.sub(r'-+', '-', re.sub(r'[^A-Za-z0-9-]', '-', str(value))).strip('-').lower()
    if not slug(prefix):
        raise PublishError('HF repo prefix must contain letters or numbers')
    model_name = str(item['base_model']).rsplit('/', 1)[-1]
    pieces = [slug(prefix)[:24], slug(model_name)[:48], slug(item['arm']),
              item['finetune_mode'], f"s{item['seed']}", f"k{item['K']}",
              'a' + slug(format(item['alpha'], '.8g')), f"e{item['epoch']:02d}"]
    identity = json.dumps({key: value for key, value in item.items() if key not in ('folder', 'files')}, sort_keys=True)
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:12]
    # Trim the human-readable prefix/model first, retaining every setting tag.
    while len('-'.join(pieces)) + 13 > 96 and (len(pieces[0]) > 1 or len(pieces[1]) > 1):
        index = 0 if len(pieces[0]) > 1 else 1
        pieces[index] = pieces[index][:-1].rstrip('-') or 'x'
    name = '-'.join(pieces) + '-' + suffix
    if len(name) > 96 or '..' in name or '--' in name:
        raise PublishError('Training setting values are too long for a Hugging Face repository name')
    return f'{namespace}/{name}'


def _head(api, repo_id, token, revision=None):
    info = api.repo_info(repo_id, repo_type='model', token=token,
                         files_metadata=True, revision=revision, timeout=30)
    sha = _field(info, 'sha')
    if not isinstance(sha, str) or not sha:
        raise PublishError('Cannot establish the repository commit; refusing an unguarded upload')
    return info


def _check_privacy(info, private):
    if _field(info, 'private') is not private:
        raise PublishError('Repository visibility differs from the required private/public policy; no files uploaded')


def _remote_matches(api, repo_id, token, info, item, hashes):
    """Verify one pinned commit, using LFS SHA256 or ordinary Git blob hashes."""
    siblings = {_field(file, 'rfilename'): file for file in _field(info, 'siblings', [])}
    if 'export_manifest.json' not in siblings:
        if set(siblings) - {'.gitattributes'}:
            raise PublishError('An existing repository contains unrelated files; refusing to overwrite it')
        return False
    if set(siblings) - {'.gitattributes'} != set(hashes):
        raise PublishError('Remote repository files differ from the checkpoint allowlist')
    path = api.hf_hub_download(repo_id, 'export_manifest.json', repo_type='model',
                               revision=_field(info, 'sha'), token=token,
                               local_files_only=False)
    try:
        manifest = json.loads(Path(path).read_text())
    except (ValueError, UnicodeError):
        raise PublishError('Remote checkpoint manifest is not valid JSON') from None
    if manifest != {key: value for key, value in item.items() if key != 'folder'}:
        raise PublishError('An existing repository contains a different checkpoint; refusing to overwrite it')
    folder = Path(item['folder'])
    for name, digest in hashes.items():
        remote = siblings[name]
        if _field(remote, 'size') != (folder / name).stat().st_size:
            raise PublishError('Remote checkpoint file size verification failed')
        lfs = _field(remote, 'lfs')
        if lfs:
            matches = _field(lfs, 'sha256') == digest
        else:
            matches = _field(remote, 'blob_id') == _git_blob(folder / name)
        if not matches:
            raise PublishError('Remote checkpoint file hash verification failed')
    return True


def _transient(exc):
    code = _field(getattr(exc, 'response', None), 'status_code')
    return (code in (408, 409, 425, 429) or isinstance(code, int) and 500 <= code < 600
            or isinstance(exc, (ConnectionError, TimeoutError))
            or type(exc).__name__ in ('ConnectionError', 'ConnectTimeout', 'ReadTimeout', 'Timeout'))


def _publish_one(api, token, repo_id, item, hashes, staging, retries, sleep_fn):
    private = private_all() or item['arm'] in PRIVATE_ARMS
    status = dict(epoch=item['epoch'], step=item['step'], repo_id=repo_id, private=private,
                  status='failed', attempts=0)
    for attempt in range(1, retries + 1):
        status['attempts'] = attempt
        try:
            # exist_ok does not change an existing repository's privacy.
            api.create_repo(repo_id, repo_type='model', token=token, private=private, exist_ok=True)
            info = _head(api, repo_id, token)
            _check_privacy(info, private)
            if _remote_matches(api, repo_id, token, info, item, hashes):
                state, commit = 'already_uploaded', _field(info, 'sha')
            else:
                # Also catches source changes between retries / preparation and upload.
                if verify_export(item, staging) != hashes:
                    raise PublishError('Checkpoint export changed before upload')
                result = api.upload_folder(repo_id=repo_id, repo_type='model', token=token,
                    folder_path=item['folder'], path_in_repo='', revision='main',
                    parent_commit=_field(info, 'sha'), allow_patterns=sorted(hashes),
                    commit_message=f"Upload verified {item['arm']} {item['finetune_mode']} epoch {item['epoch']}",
                    run_as_future=False)
                commit = _field(result, 'oid')
                if not isinstance(commit, str) or not commit:
                    raise PublishError('Upload returned no commit; retry to verify the repository')
                verified = _head(api, repo_id, token, revision=commit)
                _check_privacy(verified, private)
                if not _remote_matches(api, repo_id, token, verified, item, hashes):
                    raise PublishError('The completed upload could not be verified; retry this upload')
                state = 'uploaded'
            status.update(status=state, commit=commit,
                          commit_url=f'{HUB_ENDPOINT}/{repo_id}/commit/{commit}')
            status.pop('error', None)
            return status
        except Exception as exc:
            status['error'] = _safe_error(exc)
            if attempt == retries or not _transient(exc):
                return status
            sleep_fn(min(2 ** (attempt - 1), 30))
    return status


def _write_status(run_dir, status):
    target = Path(run_dir) / 'hf_upload_status.json'
    if target.is_symlink():
        raise PublishError('Upload status file must not be a symbolic link')
    temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def upload_run(run_dir, *, epochs=(2, 3, 4), namespace, prefix='stride-iclr2027',
               description='', retries=3, api=None, export_fn=None, sleep_fn=time.sleep):
    """Upload completed epoch exports; return status and never delete checkpoints.

    ``retries`` is the maximum number of attempts per epoch.  A repeated call
    verifies and reuses an identical remote commit rather than overwriting it.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise PublishError('Training run directory does not exist')
    namespace = _namespace(namespace)
    epochs = tuple(epochs)
    if not epochs or any(type(epoch) is not int or epoch < 1 for epoch in epochs) or len(set(epochs)) != len(epochs):
        raise PublishError('Upload epochs must be distinct positive integers')
    if type(retries) is not int or not 1 <= retries <= 10:
        raise PublishError('Upload retries must be between 1 and 10')
    _secret_guard(str(prefix) + '\n' + str(description))
    status = dict(schema_version=1, namespace=namespace, repo_prefix=prefix,
                  requested_epochs=list(epochs), complete=False, checkpoints=[])
    try:
        # Validate visibility before credentials, exports or any network call.
        status['private_all'] = private_all()
        token = _token()
        if export_fn is None:
            from hf_export import prepare_exports
            export_fn = prepare_exports
        staging = run_dir / 'hf_exports'
        exports = export_fn(run_dir, staging, epochs=epochs, description=description)
        if not exports or len({item['epoch'] for item in exports}) != len(exports):
            raise PublishError('No unique completed epoch exports were prepared')
        if any(item['epoch'] not in epochs for item in exports):
            raise PublishError('Exporter returned an epoch that was not requested')
        prepared = [(item, verify_export(item, staging)) for item in exports]
        api = _api(api, token)
        preflight_auth(namespace, token=token, api=api)
        for item, hashes in prepared:
            repo_id = repository_id(namespace, prefix, item)
            result = _publish_one(api, token, repo_id, item, hashes, staging, retries, sleep_fn)
            status['checkpoints'].append(result)
            _write_status(run_dir, status)
        status['complete'] = all(item['status'] in ('uploaded', 'already_uploaded') for item in status['checkpoints'])
    except Exception as exc:
        status['error'] = _safe_error(exc)
    _write_status(run_dir, status)
    return status


def _epochs(text):
    try:
        values = tuple(int(value.strip()) for value in text.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError('Use comma-separated epoch integers, e.g. 2,3,4') from None
    if not values or len(set(values)) != len(values) or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError('Epochs must be distinct positive integers')
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir')
    parser.add_argument('--epochs', type=_epochs, default=(2, 3, 4))
    parser.add_argument('--namespace', default=os.environ.get('HF_USERNAME', ''))
    parser.add_argument('--repo-prefix', default=os.environ.get('HF_REPO_PREFIX', 'stride-iclr2027'))
    parser.add_argument('--description', default=os.environ.get('HF_DESCRIPTION', ''))
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    if not args.preflight and not args.run_dir:
        parser.error('--run-dir is required unless --preflight is used')
    # Training intentionally uses cached/offline model loading.  Only this
    # dedicated child process is put online, before importing the Hub SDK.
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ['HF_ENDPOINT'] = HUB_ENDPOINT
    try:
        if args.preflight:
            result = preflight_auth(args.namespace)
            print(json.dumps({'hf_preflight': result}, sort_keys=True))
            return 0
        result = upload_run(args.run_dir, epochs=args.epochs, namespace=args.namespace,
                            prefix=args.repo_prefix, description=args.description, retries=args.retries)
        print(json.dumps({'hf_upload': result}, sort_keys=True))
        return 0 if result['complete'] else 1
    except Exception as exc:
        print(json.dumps({'hf_upload_error': _safe_error(exc)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
