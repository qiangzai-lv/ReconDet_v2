#!/usr/bin/env python3
"""Probe DrivingSDK's differentiable rotated IoU operator on NPU."""

import argparse
import json
import os
from typing import Any, Dict


OPERATOR_NAME = 'diff_iou_rotated_2d'


def _environment() -> Dict[str, str]:
    keys = ('ASCEND_HOME_PATH', 'ASCEND_OPP_PATH',
            'ASCEND_CUSTOM_OPP_PATH')
    return {key: os.environ.get(key, '') for key in keys}


def _run_operator() -> Dict[str, Any]:
    try:
        import torch
        import mx_driving
    except Exception as exc:
        return {
            'name': OPERATOR_NAME,
            'status': 'unavailable',
            'detail': f'{type(exc).__name__}: {exc}',
        }

    if not hasattr(torch, 'npu'):
        return {
            'name': OPERATOR_NAME,
            'status': 'unavailable',
            'detail': 'torch.npu is unavailable',
        }
    try:
        if not torch.npu.is_available():
            raise RuntimeError('torch.npu reports no available NPU device')
        operator = getattr(mx_driving, OPERATOR_NAME)
        boxes_a = torch.tensor(
            [[[0.0, 0.0, 2.0, 1.0, 0.1]]],
            device='npu', dtype=torch.float32, requires_grad=True)
        boxes_b = torch.tensor(
            [[[0.0, 0.0, 2.0, 1.0, 0.1]]],
            device='npu', dtype=torch.float32)

        iou = operator(boxes_a, boxes_b)
        if tuple(iou.shape) != (1, 1):
            raise RuntimeError(f'unexpected output shape: {tuple(iou.shape)}')
        if not bool(torch.isfinite(iou).all().item()):
            raise RuntimeError('forward returned non-finite values')
        if not bool(torch.allclose(
                iou, torch.ones_like(iou), atol=1e-4, rtol=1e-4)):
            raise RuntimeError(f'identical boxes returned IoU={iou.item():.6g}')

        iou.sum().backward()
        if boxes_a.grad is None or not bool(
                torch.isfinite(boxes_a.grad).all().item()):
            raise RuntimeError('backward produced no finite gradient')
    except Exception as exc:
        return {
            'name': OPERATOR_NAME,
            'status': 'failed',
            'detail': f'{type(exc).__name__}: {exc}',
        }
    return {
        'name': OPERATOR_NAME,
        'status': 'ok',
        'detail': 'DrivingSDK forward and backward succeeded',
    }


def probe(dry_run: bool = False) -> Dict[str, Any]:
    check = ({
        'name': OPERATOR_NAME,
        'status': 'skipped',
        'detail': 'dry run',
    } if dry_run else _run_operator())
    return {
        'status': 'skipped' if dry_run else check['status'],
        'dry_run': dry_run,
        'device': 'npu',
        'environment': _environment(),
        'checks': [check],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='show the check without importing or using NPU')
    parser.add_argument('--json', action='store_true',
                        help='emit machine-readable JSON')
    parser.add_argument('--strict', action='store_true',
                        help='return 1 unless the operator check is OK')
    args = parser.parse_args(argv)
    payload = probe(dry_run=args.dry_run)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        check = payload['checks'][0]
        detail = f": {check['detail']}" if 'detail' in check else ''
        print(f"status={payload['status']} device=npu")
        print(f"[{check['status'].upper()}] {check['name']}{detail}")
    return int(args.strict and payload['status'] != 'ok')


if __name__ == '__main__':
    raise SystemExit(main())
