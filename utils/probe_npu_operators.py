#!/usr/bin/env python3
"""Probe Ascend DrivingSDK operators required by ReconDet.

This script intentionally does not import the ReconDet model. It is safe to
run before installing DrivingSDK and reports one independent result per op.
On an Ascend host, use the same Python environment as the training command.
"""

import argparse
import importlib
import json
import os
import platform
import sys
import time
import traceback
from typing import Callable, Dict, List


CHECK_NAMES = (
    'npu_rotated_iou',
    'diff_iou_rotated_2d_forward',
    'diff_iou_rotated_2d_backward',
    'nms3d',
    'nms3d_normal',
)


def _load_environment() -> Dict[str, object]:
    info = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'torch': None,
        'torch_npu': None,
        'mx_driving': None,
        'device': 'npu',
        'device_name': None,
        'cann_path': os.environ.get('ASCEND_HOME_PATH') or os.environ.get(
            'ASCEND_TOOLKIT_HOME'),
    }
    try:
        torch = importlib.import_module('torch')
        info['torch'] = getattr(torch, '__version__', 'unknown')
    except Exception as exc:  # pragma: no cover - environment dependent
        info['torch_import_error'] = f'{type(exc).__name__}: {exc}'
        return info
    try:
        torch_npu = importlib.import_module('torch_npu')
        info['torch_npu'] = getattr(torch_npu, '__version__', 'unknown')
        npu = getattr(torch_npu, 'npu', None)
        if npu is not None and hasattr(npu, 'is_available') and npu.is_available():
            try:
                info['device_name'] = npu.get_device_name(0)
            except Exception as exc:
                info['device_name_error'] = f'{type(exc).__name__}: {exc}'
    except Exception as exc:  # pragma: no cover - environment dependent
        info['torch_npu_import_error'] = f'{type(exc).__name__}: {exc}'
    try:
        mx_driving = importlib.import_module('mx_driving')
        info['mx_driving'] = getattr(mx_driving, '__version__', 'installed')
    except Exception as exc:  # pragma: no cover - environment dependent
        info['mx_driving_import_error'] = f'{type(exc).__name__}: {exc}'
    return info


def _result(name: str, status: str, started: float, **kwargs) -> Dict[str, object]:
    item = {
        'name': name,
        'status': status,
        'elapsed_ms': round((time.perf_counter() - started) * 1000.0, 3),
    }
    item.update(kwargs)
    return item


def _get_op(mx_driving, name: str):
    op = getattr(mx_driving, name, None)
    if op is not None:
        return op, f'mx_driving.{name}'
    detection = getattr(mx_driving, 'detection', None)
    op = getattr(detection, name, None) if detection is not None else None
    if op is not None:
        return op, f'mx_driving.detection.{name}'
    return None, None


def _npu_tensor(torch, value):
    return torch.tensor(value, dtype=torch.float32, device='npu')


def _run_check(name: str, mx_driving, torch) -> Dict[str, object]:
    started = time.perf_counter()
    op_name = name.removesuffix('_forward').removesuffix('_backward')
    op, qualified_name = _get_op(mx_driving, op_name)
    if op is None:
        return _result(
            name, 'unavailable', started,
            error=f'{op_name} is not exported by mx_driving or mx_driving.detection')

    try:
        if name == 'npu_rotated_iou':
            # [x, y, w, h, angle], angle is in degrees for this beta API.
            boxes_a = _npu_tensor(torch, [[[0., 0., 2., 1., 0.]]])
            boxes_b = _npu_tensor(torch, [[[0.1, 0., 2., 1., 10.]]])
            output = op(boxes_a, boxes_b, False, 0, True, 1e-5, 1e-5)
            expected_shape = (1, 1, 1)
        elif name == 'diff_iou_rotated_2d_forward':
            # [x_center, y_center, dx, dy, angle], angle is in radians.
            boxes_a = _npu_tensor(torch, [[[0., 0., 2., 1., 0.]]])
            boxes_b = _npu_tensor(torch, [[[0.1, 0., 2., 1., 0.1]]])
            output = op(boxes_a, boxes_b)
            expected_shape = (1, 1)
        elif name == 'diff_iou_rotated_2d_backward':
            boxes_a = _npu_tensor(torch, [[[0., 0., 2., 1., 0.]]])
            boxes_b = _npu_tensor(torch, [[[0.1, 0., 2., 1., 0.1]]])
            boxes_a.requires_grad_(True)
            output = op(boxes_a, boxes_b)
            output.sum().backward()
            if boxes_a.grad is None:
                raise RuntimeError('operator returned no gradient for boxes_a')
            if not torch.isfinite(boxes_a.grad).all().item():
                raise FloatingPointError('non-finite gradient')
            expected_shape = (1, 1)
        elif name in ('nms3d', 'nms3d_normal'):
            boxes = _npu_tensor(torch, [
                [0., 0., 0., 2., 1., 1., 0.],
                [0.1, 0., 0., 2., 1., 1., 0.],
            ])
            scores = _npu_tensor(torch, [0.9, 0.8])
            output = op(boxes, scores, 0.5)
            expected_shape = (None,)
        else:  # pragma: no cover
            raise ValueError(f'unknown check {name}')

        if output is None:
            raise RuntimeError('operator returned None')
        shape = tuple(output.shape)
        if expected_shape != (None,) and shape != expected_shape:
            raise RuntimeError(
                f'unexpected output shape {shape}, expected {expected_shape}')
        return _result(
            name, 'ok', started,
            implementation=qualified_name,
            output_shape=list(shape),
            output_dtype=str(output.dtype),
            output_device=str(output.device))
    except Exception as exc:  # pragma: no cover - hardware dependent
        return _result(
            name, 'failed', started,
            implementation=qualified_name,
            error=f'{type(exc).__name__}: {exc}',
            traceback=traceback.format_exc())


def probe(dry_run: bool = False) -> Dict[str, object]:
    environment = _load_environment()
    if dry_run:
        checks = [
            _result(name, 'not_run', time.perf_counter(), reason='dry-run')
            for name in CHECK_NAMES
        ]
        status = 'ok'
    elif environment.get('mx_driving') is None:
        reason = environment.get(
            'mx_driving_import_error', 'mx_driving is unavailable')
        checks = [
            _result(name, 'unavailable', time.perf_counter(), error=str(reason))
            for name in CHECK_NAMES
        ]
        status = 'unavailable'
    elif environment.get('torch_npu') is None:
        checks = [
            _result(
                name, 'unavailable', time.perf_counter(),
                error='torch_npu is unavailable')
            for name in CHECK_NAMES
        ]
        status = 'unavailable'
    else:
        torch = importlib.import_module('torch')
        mx_driving = importlib.import_module('mx_driving')
        checks = [_run_check(name, mx_driving, torch) for name in CHECK_NAMES]
        statuses = {item['status'] for item in checks}
        status = 'ok' if statuses == {'ok'} else 'failed'
    return {
        'status': status,
        'dry_run': dry_run,
        'environment': environment,
        'device': 'npu',
        'checks': checks,
    }


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='emit JSON only')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate the probe contract without importing NPU ops')
    parser.add_argument('--strict', action='store_true',
                        help='return non-zero when any check is unavailable or fails')
    args = parser.parse_args(argv)
    payload = probe(dry_run=args.dry_run)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"status={payload['status']} device={payload['device']}")
        print(json.dumps(payload['environment'], ensure_ascii=False, indent=2))
        for item in payload['checks']:
            suffix = item.get('implementation', '')
            error = item.get('error', '')
            print(f"[{item['status']}] {item['name']} {suffix} {error}".rstrip())
    if args.strict and payload['status'] != 'ok':
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
