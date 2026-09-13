#!/usr/bin/env python3
"""Diagnose CANN/torch_npu/DrivingSDK runtime compatibility."""

import argparse
import ctypes.util
import importlib
import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path


SYMBOL = 'aclnnDiffIouRotatedSortVertices'


def _command(command):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return {'error': str(exc)}
    return {
        'returncode': result.returncode,
        'stdout': result.stdout.strip(),
        'stderr': result.stderr.strip(),
    }


def _module_info(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {'name': name, 'available': False}
    info = {'name': name, 'available': True, 'origin': spec.origin}
    try:
        module = importlib.import_module(name)
        info['version'] = getattr(module, '__version__', 'unknown')
    except Exception as exc:  # pragma: no cover - hardware dependent
        info['import_error'] = f'{type(exc).__name__}: {exc}'
    return info


def _candidate_libs():
    roots = []
    for key in ('ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME'):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value))
    for value in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep):
        if value:
            roots.append(Path(value))
    roots.extend((Path('/usr/local/Ascend'), Path('/usr/local/Ascend/latest')))
    found = []
    for root in roots:
        if root.name == 'lib64' and root.name:
            candidate = root / 'libopapi.so'
            if candidate.is_file():
                found.append(candidate)
            continue
        if root.is_dir():
            for candidate in root.glob('**/libopapi.so'):
                if candidate.is_file():
                    found.append(candidate)
    return sorted(set(found))


def diagnose():
    payload = {
        'platform': platform.platform(),
        'python': platform.python_version(),
        'environment': {
            key: os.environ.get(key)
            for key in ('ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME',
                        'LD_LIBRARY_PATH')
        },
        'modules': [_module_info(name) for name in
                    ('torch', 'torch_npu', 'mx_driving')],
        'libopapi_lookup': ctypes.util.find_library('opapi'),
        'libopapi_candidates': [],
    }
    for path in _candidate_libs():
        item = {'path': str(path)}
        item['symbol_check'] = _command(
            ['nm', '-D', '--defined-only', str(path)])
        symbols = item['symbol_check'].get('stdout', '')
        item['has_diff_iou_sort_symbol'] = any(
            SYMBOL in line for line in symbols.splitlines())
        item['version_info'] = _command(
            ['strings', str(path)])
        item['version_info']['stdout'] = '\n'.join(
            line for line in item['version_info'].get('stdout', '').splitlines()
            if 'CANN' in line or 'version' in line.lower())[:2000]
        payload['libopapi_candidates'].append(item)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    payload = diagnose()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if args.json or payload['modules'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
