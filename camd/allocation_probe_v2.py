"""Inspect a one-GPU Slurm allocation and test only its bounded GPU masks.

No model, vLLM, permission change, scheduler mutation, or unallocated-device
enumeration by HIP is performed. GPU operations require exactly one visible
device. Device files are opened and immediately closed only to inspect access.
"""
import errno
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

ROOT = Path('/path/to/cluster-user/system/stride_sample100_20260915/results')
ENV_NAMES = ['SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURMD_NODENAME',
             'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES',
             'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL',
             'PYTORCH_NVML_BASED_CUDA_CHECK']
MASK_NAMES = ['ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
              'HIP_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL']


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def number(text):
    try:
        return int(text, 0) if text.lower().startswith('0x') else int(text)
    except (ValueError, AttributeError):
        return None


def device_inventory():
    paths = [Path('/dev/kfd')]
    paths += sorted(Path('/dev/dri').glob('card*'))
    paths += sorted(Path('/dev/dri').glob('renderD*'))
    result = []
    for path in paths:
        item = {'path': str(path), 'open_read_write': False}
        try:
            info = path.stat()
            item.update(mode=stat.filemode(info.st_mode), uid=info.st_uid,
                        gid=info.st_gid, character_device=stat.S_ISCHR(info.st_mode),
                        major=os.major(info.st_rdev), minor=os.minor(info.st_rdev),
                        resolved_path=str(path.resolve()))
            if not stat.S_ISCHR(info.st_mode):
                raise ValueError('Expected a character device; no open attempted')
            descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NONBLOCK)
            os.close(descriptor)
            item['open_read_write'] = True
        except OSError as error:
            item.update(error_type=type(error).__name__, errno=error.errno,
                        errno_name=errno.errorcode.get(error.errno))
        except ValueError as error:
            item.update(error_type=type(error).__name__, error=str(error))
        result.append(item)
    return result


def topology_inventory():
    result = []
    for node in sorted(Path('/sys/class/kfd/kfd/topology/nodes').glob('*')):
        if not node.name.isdecimal():
            continue
        item = {'node': node.name, 'properties': {}}
        for name in ('gpu_id', 'name', 'properties'):
            try:
                text = (node / name).read_text()[:16384].strip()
                if name == 'properties':
                    item[name] = dict(line.split(None, 1) for line in text.splitlines()
                                      if len(line.split(None, 1)) == 2)
                else:
                    item[name] = text
            except OSError as error:
                item[name + '_error'] = {'errno': error.errno, 'type': type(error).__name__}
        result.append(item)
    return result


def readable_gres():
    result = {}
    for filename in ('/etc/slurm/gres.conf', '/etc/slurm-llnl/gres.conf',
                     '/etc/slurm/gres.conf.d/gpu.conf'):
        path = Path(filename)
        try:
            # GRES device configuration only. Do not read slurm.conf, which can
            # contain unrelated accounting credentials.
            with path.open() as stream:
                text = stream.read(65537)
            result[filename] = {'content': text[:65536], 'truncated': len(text) > 65536}
        except OSError as error:
            result[filename] = {'errno': error.errno, 'type': type(error).__name__}
    return result


CHILD = r'''
import json, os, sys
from pathlib import Path
r = {'passed': False, 'environment': {k: os.environ.get(k) for k in
     ['ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES',
      'GPU_DEVICE_ORDINAL','PYTORCH_NVML_BASED_CUDA_CHECK']}}
target = Path(sys.argv[1])
target.write_text(json.dumps(r) + '\n')
try:
    import torch
    r['torch_version'] = torch.__version__
    r['hip_version'] = torch.version.hip
    if not torch.version.hip:
        raise ValueError('Expected ROCm PyTorch')
    r['available'] = torch.cuda.is_available()
    r['count'] = torch.cuda.device_count()
    if r['count'] != 1:
        raise ValueError('Refusing GPU allocation: visible device count is not one')
    value = torch.arange(16, device='cuda:0', dtype=torch.float32)
    torch.cuda.synchronize(0)
    r['sum'] = float(value.sum())
    if r['sum'] != 120.0:
        raise ValueError('Incorrect tensor result')
    properties = torch.cuda.get_device_properties(0)
    r['gpu'] = {'name': properties.name,
                'gcn_arch': getattr(properties, 'gcnArchName', None),
                'uuid': str(getattr(properties, 'uuid', '')),
                'total_memory': properties.total_memory}
    r['passed'] = True
except BaseException as error:
    r.update(passed=False, error_type=type(error).__name__, error=str(error))
finally:
    target.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r), flush=True)
'''


def main():
    job, step = os.environ.get('SLURM_JOB_ID'), os.environ.get('SLURM_STEP_ID')
    physical = os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS')
    if not job or not step or physical is None or not physical.isdecimal():
        raise ValueError('Requires one numeric scheduler GPU inside one allocated Slurm step')
    original = os.environ.get('ROCR_VISIBLE_DEVICES', '')
    if not (original.isdecimal() or re.fullmatch(r'GPU-[0-9a-fA-F]+', original)):
        raise ValueError('Original ROCR mask must identify one GPU')
    directory = ROOT / ('allocation-probe-v2-' + job)
    directory.mkdir(exist_ok=False)
    report = {'passed': False, 'complete': False, 'environment':
              {name: os.environ.get(name) for name in ENV_NAMES},
              'identity': {'uid': os.getuid(), 'gid': os.getgid(), 'groups': os.getgroups()},
              'variants': {}}
    target = directory / 'report.json'
    write(target, report)
    try:
        report['devices'] = device_inventory()
        report['topology'] = topology_inventory()
        report['gres_configuration'] = readable_gres()
        variants = [('scheduler_rocr_only', original, None),
                    ('step_physical_rocr_only', physical, None)]
        # This tests whether this runtime interprets the second mask physically.
        # Both selectors are still the single scheduler-assigned GPU number.
        variants.append(('step_physical_both_masks', physical, physical))
        permitted = [item for item in report['devices']
                     if Path(item['path']).name.startswith('renderD')
                     and item['open_read_write']]
        report['permitted_render_count'] = len(permitted)
        if len(permitted) == 1:
            allowed_minor = permitted[0]['minor']
            matches = [node for node in report['topology']
                       if number(node['properties'].get('drm_render_minor')) == allowed_minor
                       and number(node.get('gpu_id')) not in (None, 0)]
            report['permitted_render_topology'] = matches
            if len(matches) == 1:
                unique = number(matches[0]['properties'].get('unique_id'))
                if unique is not None and 0 < unique < 2**64:
                    uuid = 'GPU-' + format(unique, '016x')
                    report['derived_allowed_gpu_uuid'] = uuid
                    variants.append(('permitted_render_uuid_only', uuid, None))
                    variants.append(('permitted_render_uuid_logical_hip', uuid, '0'))
                else:
                    report['uuid_probe_skipped'] = 'No valid 64-bit unique_id for permitted render node'
            else:
                report['uuid_probe_skipped'] = 'Permitted render node did not have exactly one GPU topology match'
        else:
            report['uuid_probe_skipped'] = 'Cannot derive assigned UUID without exactly one permitted render node'
        write(target, report)
        for label, rocr, hip in variants:
            env = os.environ.copy()
            for name in MASK_NAMES:
                env.pop(name, None)
            env.pop('PYTORCH_NVML_BASED_CUDA_CHECK', None)
            env['ROCR_VISIBLE_DEVICES'] = rocr
            if hip is not None:
                env.update(HIP_VISIBLE_DEVICES=hip, CUDA_VISIBLE_DEVICES=hip)
            output = directory / (label + '.json')
            started = time.monotonic()
            try:
                process = subprocess.run([sys.executable, '-s', '-c', CHILD, str(output)],
                                         env=env, check=False, timeout=60,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                result = json.loads(output.read_text()) if output.is_file() else {
                    'passed': False, 'error': 'Missing explicit child completion artifact'}
                result['returncode_informational_only'] = process.returncode
                result['stderr'] = process.stderr[-12000:]
            except subprocess.TimeoutExpired:
                result = {'passed': False, 'error': 'Child timed out after 60 seconds'}
            except (OSError, ValueError) as error:
                result = {'passed': False, 'error_type': type(error).__name__, 'error': str(error)}
            result['elapsed_seconds'] = time.monotonic() - started
            report['variants'][label] = result
            write(target, report)
            print('ALLOCATION_VARIANT', label, json.dumps(result), flush=True)
        report['complete'] = True
        report['passed'] = any(item.get('passed') is True for item in report['variants'].values())
    except BaseException as error:
        report.update(passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(target, report)
        print('ALLOCATION_PROBE_V2', json.dumps({'complete': report['complete'],
              'passed': report['passed'], 'report': str(target)}), flush=True)


if __name__ == '__main__':
    main()
