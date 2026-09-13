#!/usr/bin/env python3
"""Check whether DrivingSDK diff_iou_rotated_2d supports autograd."""

import argparse
import json
import os
from typing import Any, Dict, List


OPERATOR_NAME = 'diff_iou_rotated_2d'
CHECK_NAMES = (
    'requires_grad', 'backward', 'gradient_present', 'gradient_finite')


def _environment() -> Dict[str, str]:
    keys = ('ASCEND_HOME_PATH', 'ASCEND_OPP_PATH',
            'ASCEND_CUSTOM_OPP_PATH')
    return {key: os.environ.get(key, '') for key in keys}


def _result(name: str, status: str, detail: str = '') -> Dict[str, str]:
    item = {'name': name, 'status': status}
    if detail:
        item['detail'] = detail
    return item


def _run() -> Dict[str, Any]:
    checks: List[Dict[str, str]] = []
    try:
        import torch
        import mx_driving
    except Exception as exc:
        detail = f'{type(exc).__name__}: {exc}'
        checks = [_result(name, 'unavailable', detail) for name in CHECK_NAMES]
        return {'status': 'unavailable', 'detail': detail, 'checks': checks}

    try:
        if not hasattr(torch, 'npu') or not torch.npu.is_available():
            raise RuntimeError('torch.npu reports no available NPU device')
        operator = getattr(mx_driving, OPERATOR_NAME)
        boxes_a = torch.tensor(
            [[[0.15, 0.05, 2.0, 1.2, 0.23]]],
            device='npu', dtype=torch.float32, requires_grad=True)
        boxes_b = torch.tensor(
            [[[-0.10, 0.0, 1.8, 1.0, -0.11]]],
            device='npu', dtype=torch.float32)
        output = operator(boxes_a, boxes_b)
        requires_grad = bool(output.requires_grad)
        checks.append(_result(
            'requires_grad', 'ok' if requires_grad else 'failed',
            f'output.requires_grad={requires_grad}'))
        if not requires_grad:
            raise RuntimeError('operator output does not require gradients')

        try:
            output.sum().backward()
        except Exception as exc:
            checks.append(_result(
                'backward', 'failed', f'{type(exc).__name__}: {exc}'))
            for name in ('gradient_present', 'gradient_finite'):
                checks.append(_result(name, 'skipped', 'backward failed'))
            return {
                'status': 'failed',
                'detail': f'{type(exc).__name__}: {exc}',
                'checks': checks,
            }
        checks.append(_result('backward', 'ok'))

        gradient = boxes_a.grad
        present = gradient is not None
        checks.append(_result(
            'gradient_present', 'ok' if present else 'failed',
            f'boxes_a.grad is not None: {present}'))
        finite = present and bool(torch.isfinite(gradient).all().item())
        checks.append(_result(
            'gradient_finite', 'ok' if finite else 'failed',
            'gradient contains only finite values' if finite
            else 'gradient is missing or contains non-finite values'))
    except Exception as exc:
        detail = f'{type(exc).__name__}: {exc}'
        missing = {item['name'] for item in checks}
        for name in CHECK_NAMES:
            if name not in missing:
                checks.append(_result(name, 'failed', detail))
        return {'status': 'failed', 'detail': detail, 'checks': checks}

    status = 'ok' if all(item['status'] == 'ok' for item in checks) else 'failed'
    detail = 'forward/backward and finite gradient succeeded'
    return {'status': status, 'detail': detail, 'checks': checks}


def probe(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {
            'operator': OPERATOR_NAME,
            'status': 'skipped',
            'detail': 'dry run',
            'device': 'npu',
            'dry_run': True,
            'environment': _environment(),
            'checks': list(CHECK_NAMES),
        }
    payload = _run()
    payload.update({
        'operator': OPERATOR_NAME,
        'device': 'npu',
        'dry_run': False,
        'environment': _environment(),
    })
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args(argv)
    payload = probe(args.dry_run)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status={payload['status']} operator={payload['operator']}")
        for item in payload['checks']:
            if isinstance(item, str):
                print(f'[PENDING] {item}')
            else:
                detail = f": {item['detail']}" if 'detail' in item else ''
                print(f"[{item['status'].upper()}] {item['name']}{detail}")
    return int(args.strict and payload['status'] != 'ok')


if __name__ == '__main__':
    raise SystemExit(main())
