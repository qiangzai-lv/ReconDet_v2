#!/usr/bin/env python3
"""Probe the NPU operators used by the ReConDet geometry path.

The probe deliberately executes a tiny input on NPU.  Importing ``mx_driving``
alone only proves that the Python extension is installed; it does not prove
that the corresponding CANN/OPP kernel is registered and runnable.
"""

import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List


CHECK_NAMES = (
    'npu_rotated_iou',
    'diff_iou_rotated_2d_forward',
    'diff_iou_rotated_2d_backward',
    'nms3d',
    'nms3d_normal',
)
RUNTIME_OPERATOR_NAME = 'aclnnDiffIouRotatedSortVertices'


def _check(name: str, status: str, detail: str = '') -> Dict[str, str]:
    result = {'name': name, 'status': status}
    if detail:
        result['detail'] = detail
    return result


def _environment() -> Dict[str, str]:
    keys = ('ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'ASCEND_CUSTOM_OPP_PATH')
    return {key: os.environ.get(key, '') for key in keys}


def _libopapi_candidates() -> List[Path]:
    candidates = []
    for key in ('ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME'):
        root = os.environ.get(key)
        if root:
            root = Path(root)
            candidates.extend((root / 'lib64' / 'libopapi.so',
                               root / 'runtime' / 'lib64' / 'libopapi.so'))
    candidates.extend((
        Path('/usr/local/Ascend/ascend-toolkit/latest/lib64/libopapi.so'),
        Path('/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/libopapi.so'),
        Path('/usr/local/Ascend/latest/lib64/libopapi.so'),
    ))
    discovered = ctypes.util.find_library('opapi')
    if discovered:
        candidates.append(Path(discovered))
    unique = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _probe_runtime_operator() -> Dict[str, str]:
    """Check the ACLNN symbol before importing or executing model code."""
    candidates = _libopapi_candidates()
    existing = [path for path in candidates if path.is_file()]
    # ``find_library`` can return a soname resolved only by the dynamic
    # loader, so retain it as a load candidate even when it is not a file.
    load_candidates = [str(path) for path in candidates if not path.is_absolute()]
    existing_names = [str(path) for path in existing] + load_candidates
    if not existing_names:
        return {
            'name': RUNTIME_OPERATOR_NAME,
            'status': 'unavailable',
            'detail': 'libopapi.so was not found in configured/common paths',
        }
    load_errors = []
    for path in existing_names:
        try:
            library = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            load_errors.append(f'{path}: {exc}')
            continue
        if getattr(library, RUNTIME_OPERATOR_NAME, None) is not None:
            return {
                'name': RUNTIME_OPERATOR_NAME,
                'status': 'ok',
                'library': path,
            }
        load_errors.append(f'{path}: symbol not exported')
    return {
        'name': RUNTIME_OPERATOR_NAME,
        'status': 'failed',
        'detail': '; '.join(load_errors),
    }


def _load_runtime():
    import torch
    import mx_driving

    if not hasattr(torch, 'npu'):
        raise RuntimeError('torch.npu is unavailable in this Python environment')
    if not torch.npu.is_available():
        raise RuntimeError('torch.npu reports no available NPU device')
    return torch, mx_driving


def _operator(module: Any, name: str) -> Callable:
    operator = getattr(module, name, None)
    if operator is None:
        raise AttributeError(f'mx_driving.{name} is not exported')
    return operator


def _run_npu_rotated_iou(torch, mx_driving) -> None:
    operator = _operator(mx_driving, 'npu_rotated_iou')
    boxes_a = torch.tensor(
        [[[0.0, 0.0, 2.0, 1.0, 0.0]]], device='npu')
    boxes_b = torch.tensor(
        [[[0.0, 0.0, 2.0, 1.0, 0.0]]], device='npu')
    output = operator(boxes_a, boxes_b, False, 0, True, 1e-5, 1e-5)
    if tuple(output.shape) != (1, 1, 1):
        raise RuntimeError(f'unexpected output shape: {tuple(output.shape)}')


def _run_diff_iou_forward(torch, mx_driving) -> None:
    operator = _operator(mx_driving, 'diff_iou_rotated_2d')
    boxes_a = torch.tensor(
        [[[0.0, 0.0, 2.0, 1.0, 0.0]]], device='npu')
    boxes_b = torch.tensor(
        [[[0.0, 0.0, 2.0, 1.0, 0.0]]], device='npu')
    output = operator(boxes_a, boxes_b)
    if tuple(output.shape) != (1, 1):
        raise RuntimeError(f'unexpected output shape: {tuple(output.shape)}')
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError('operator returned non-finite values')


def _run_diff_iou_backward(torch, mx_driving) -> None:
    operator = _operator(mx_driving, 'diff_iou_rotated_2d')
    boxes_a = torch.tensor(
        [[[0.0, 0.0, 2.0, 1.0, 0.1]]], device='npu', requires_grad=True)
    boxes_b = torch.tensor(
        [[[0.1, 0.0, 2.0, 1.0, 0.0]]], device='npu')
    loss = operator(boxes_a, boxes_b).sum()
    loss.backward()
    if boxes_a.grad is None or not bool(torch.isfinite(boxes_a.grad).all().item()):
        raise RuntimeError('backward produced no finite gradient')


def _run_nms(torch, mx_driving, name: str) -> None:
    operator = _operator(mx_driving, name)
    boxes = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0],
        [0.1, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0],
    ], device='npu')
    scores = torch.tensor([0.9, 0.5], device='npu')
    output = operator(boxes, scores, 0.25)
    if not isinstance(output, torch.Tensor):
        raise RuntimeError(f'unexpected output type: {type(output).__name__}')


def _run_checks() -> List[Dict[str, str]]:
    try:
        torch, mx_driving = _load_runtime()
    except Exception as exc:  # Environment failures affect every check.
        detail = f'{type(exc).__name__}: {exc}'
        return [_check(name, 'unavailable', detail) for name in CHECK_NAMES]

    runners = {
        'npu_rotated_iou': lambda: _run_npu_rotated_iou(torch, mx_driving),
        'diff_iou_rotated_2d_forward': (
            lambda: _run_diff_iou_forward(torch, mx_driving)),
        'diff_iou_rotated_2d_backward': (
            lambda: _run_diff_iou_backward(torch, mx_driving)),
        'nms3d': lambda: _run_nms(torch, mx_driving, 'nms3d'),
        'nms3d_normal': lambda: _run_nms(torch, mx_driving, 'nms3d_normal'),
    }
    results = []
    for name in CHECK_NAMES:
        try:
            runners[name]()
        except Exception as exc:
            results.append(_check(
                name, 'failed', f'{type(exc).__name__}: {exc}'))
        else:
            results.append(_check(name, 'ok'))
    return results


def probe(dry_run: bool = False) -> Dict[str, Any]:
    checks = ([_check(name, 'skipped', 'dry run') for name in CHECK_NAMES]
              if dry_run else _run_checks())
    runtime_operator = {
        'name': RUNTIME_OPERATOR_NAME,
        'status': 'skipped',
        'detail': 'dry run',
    } if dry_run else _probe_runtime_operator()
    if dry_run:
        return {
            'status': 'skipped',
            'dry_run': True,
            'device': 'npu',
            'environment': _environment(),
            'runtime_operator': runtime_operator,
            'checks': checks,
        }
    statuses = {item['status'] for item in checks}
    if statuses == {'ok'} and runtime_operator['status'] == 'ok':
        status = 'ok'
    elif (statuses == {'unavailable'}
          and runtime_operator['status'] == 'unavailable'):
        status = 'unavailable'
    else:
        status = 'failed'
    return {
        'status': status,
        'dry_run': dry_run,
        'device': 'npu',
        'environment': _environment(),
        'runtime_operator': runtime_operator,
        'checks': checks,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='list checks without importing or executing NPU')
    parser.add_argument('--json', action='store_true',
                        help='emit machine-readable JSON')
    parser.add_argument('--strict', action='store_true',
                        help='return 1 when any check is not OK')
    args = parser.parse_args(argv)
    payload = probe(dry_run=args.dry_run)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status={payload['status']} device=npu dry_run={payload['dry_run']}")
        for item in payload['checks']:
            detail = f": {item['detail']}" if 'detail' in item else ''
            print(f"[{item['status'].upper()}] {item['name']}{detail}")
    return int(args.strict and payload['status'] != 'ok')


if __name__ == '__main__':
    raise SystemExit(main())
