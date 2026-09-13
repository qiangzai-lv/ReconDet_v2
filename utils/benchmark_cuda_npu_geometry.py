#!/usr/bin/env python3
"""Compare ReconDet's 7DoF geometry paths on CUDA and Ascend NPU.

Run once on each device and compare the resulting JSON files:

  python utils/benchmark_cuda_npu_geometry.py --device cuda --output cuda.json
  python utils/benchmark_cuda_npu_geometry.py --device npu --output npu.json
  python utils/benchmark_cuda_npu_geometry.py --compare cuda.json npu.json

The NPU implementation uses official DrivingSDK operators. No ReconDet model
or dataset is loaded, so this isolates geometry/operator compatibility.
"""

import argparse
import hashlib
import importlib
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


OPERATIONS = ['pairwise_3d_iou', 'matched_3d_iou_backward', 'classwise_nms']


def summarize_array(value) -> Dict[str, object]:
    """Return a JSON-safe numeric summary for a tensor/array/list."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().float().numpy()
    else:
        import numpy as np
        value = np.asarray(value, dtype=np.float32)
    import numpy as np
    finite = bool(np.isfinite(value).all())
    result = {
        'shape': list(value.shape),
        'finite': finite,
        'numel': int(value.size),
        'max_abs': float(np.max(np.abs(value))) if value.size else 0.0,
        'min': float(np.min(value)) if value.size else 0.0,
        'max': float(np.max(value)) if value.size else 0.0,
        'mean': float(np.mean(value)) if value.size else 0.0,
        'values': value.reshape(-1).tolist(),
    }
    return result


def _numeric_error(left: Sequence[float], right: Sequence[float], atol: float,
                   rtol: float) -> Dict[str, object]:
    import numpy as np
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return {
            'exact': False,
            'shape_left': list(a.shape),
            'shape_right': list(b.shape),
            'max_abs_error': None,
            'max_rel_error': None,
        }
    delta = np.abs(a - b)
    # A relative error against a near-zero reference is not informative and
    # can become millions due to floating-point noise. Use the same absolute
    # tolerance as a denominator floor and expose how many entries were below
    # that floor.
    denominator = np.maximum(np.abs(b), atol)
    relative = delta / denominator
    meaningful = np.abs(b) > atol
    close = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))
    return {
        'exact': close,
        'shape': list(a.shape),
        'max_abs_error': round(float(delta.max()) if delta.size else 0.0, 12),
        'max_rel_error': round(float(relative.max()) if relative.size else 0.0, 12),
        'max_rel_error_meaningful': round(
            float(relative[meaningful].max()) if meaningful.any() else 0.0, 12),
        'relative_ignored': int((~meaningful).sum()),
    }


def compare_payloads(cuda: Dict[str, object], npu: Dict[str, object],
                     atol: float = 1e-4,
                     rtol: float = 1e-4) -> Dict[str, object]:
    """Compare saved benchmark payloads without requiring either device."""
    reports = {}
    left_inputs = cuda.get('inputs', {})
    right_inputs = npu.get('inputs', {})
    if left_inputs or right_inputs:
        reports['inputs'] = {
            'exact': left_inputs == right_inputs,
        }
    all_names = sorted(set(cuda.get('results', {})) | set(npu.get('results', {})))
    for name in all_names:
        left = cuda.get('results', {}).get(name)
        right = npu.get('results', {}).get(name)
        if left is None or right is None:
            reports[name] = {'exact': False, 'error': 'missing result'}
            continue
        if 'values' in left and 'values' in right:
            reports[name] = _numeric_error(
                left['values'], right['values'], atol, rtol)
            if 'gradient' in left and 'gradient' in right:
                reports[name]['gradient'] = _numeric_error(
                    left['gradient']['values'], right['gradient']['values'],
                    atol, rtol)
        elif 'indices' in left and 'indices' in right:
            if isinstance(left['indices'], dict) and isinstance(
                    right['indices'], dict):
                classes = sorted(set(left['indices']) | set(right['indices']))
                per_class = {}
                for cls in classes:
                    left_cls = left['indices'].get(cls, [])
                    right_cls = right['indices'].get(cls, [])
                    per_class[cls] = {
                        'exact': sorted(left_cls) == sorted(right_cls),
                        'exact_order': left_cls == right_cls,
                        'left_count': len(left_cls),
                        'right_count': len(right_cls),
                        'intersection_count': len(
                            set(left_cls).intersection(right_cls)),
                        'symmetric_difference_count': len(
                            set(left_cls).symmetric_difference(right_cls)),
                    }
                reports[name] = {
                    'exact': all(x['exact'] for x in per_class.values()),
                    'per_class': per_class,
                }
            else:
                left_indices = sorted(left['indices'])
                right_indices = sorted(right['indices'])
                reports[name] = {
                    'exact': left_indices == right_indices,
                    'exact_order': left['indices'] == right['indices'],
                    'left_count': len(left['indices']),
                    'right_count': len(right['indices']),
                    'intersection_count': len(
                        set(left['indices']).intersection(right['indices'])),
                    'symmetric_difference_count': len(
                        set(left['indices']).symmetric_difference(right['indices'])),
                }
        else:
            reports[name] = {'exact': False, 'error': 'incompatible result format'}
        left_timing = left.get('timing', {})
        right_timing = right.get('timing', {})
        if left_timing and right_timing:
            cuda_ms = left_timing.get('median_ms')
            npu_ms = right_timing.get('median_ms')
            timing_report = {}
            if cuda_ms is not None and npu_ms is not None and cuda_ms > 0:
                timing_report.update({
                    'cuda_median_ms': cuda_ms,
                    'npu_median_ms': npu_ms,
                    'npu_over_cuda': round(npu_ms / cuda_ms, 4),
                })
            cuda_backward = left_timing.get('backward_median_ms')
            npu_backward = right_timing.get('backward_median_ms')
            if (cuda_backward is not None and npu_backward is not None
                    and cuda_backward > 0):
                timing_report.update({
                    'cuda_backward_median_ms': cuda_backward,
                    'npu_backward_median_ms': npu_backward,
                    'npu_backward_over_cuda': round(
                        npu_backward / cuda_backward, 4),
                })
            if timing_report:
                reports[name]['timing'] = timing_report
    status = 'ok'
    for item in reports.values():
        if not item.get('exact', False):
            status = 'failed'
        gradient = item.get('gradient')
        if gradient is not None and not gradient.get('exact', False):
            status = 'failed'
    return {
        'status': status,
        'atol': atol,
        'rtol': rtol,
        'results': reports,
    }


def _sync(torch, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'npu':
        torch.npu.synchronize()


def _timed(torch, device, fn, warmup: int, repeats: int):
    for _ in range(warmup):
        fn()
    _sync(torch, device)
    times = []
    output = None
    for _ in range(repeats):
        _sync(torch, device)
        start = time.perf_counter()
        output = fn()
        _sync(torch, device)
        times.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(times)
    return output, {
        'mean_ms': round(statistics.mean(times), 4),
        'median_ms': round(statistics.median(times), 4),
        'p95_ms': round(ordered[min(len(ordered) - 1,
                                  math.ceil(len(ordered) * 0.95) - 1)], 4),
        'min_ms': round(min(times), 4),
        'max_ms': round(max(times), 4),
        'repeats': repeats,
        'warmup': warmup,
    }


def _build_inputs(torch, device):
    # Generate on CPU so CUDA and NPU consume byte-identical inputs. Device
    # RNG implementations are not required to produce the same sequence.
    cpu = torch.device('cpu')
    generator = torch.Generator(device=cpu).manual_seed(20260913)
    # Deliberately include overlap, no-overlap, angle wrap and unequal sizes.
    pred = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 1.0, 1.0, math.pi - 0.01],
        [0.2, 0.1, 0.1, 1.8, 1.1, 1.2, -math.pi + 0.01],
        [3.0, 0.0, 0.0, 1.0, 2.0, 1.5, 0.7],
        [0.0, 3.0, 0.0, 2.0, 2.0, 0.5, -1.2],
        [8.0, 8.0, 0.0, 0.4, 0.6, 0.8, 2.1],
        [-2.0, -1.0, 0.2, 3.0, 0.7, 2.0, -2.7],
    ], dtype=torch.float32, device=cpu)
    gt = torch.tensor([
        [0.05, 0.0, 0.0, 2.0, 1.0, 1.0, -math.pi + 0.01],
        [2.8, 0.1, 0.0, 1.2, 1.8, 1.4, 0.7],
        [0.0, 3.0, 0.1, 2.0, 2.0, 0.5, -1.2],
        [20.0, 20.0, 0.0, 1.0, 1.0, 1.0, 0.0],
    ], dtype=torch.float32, device=cpu)
    random_pred = torch.empty((26, 7), device=cpu)
    random_pred.uniform_(-6.0, 6.0, generator=generator)
    random_pred[:, 2].uniform_(-1.0, 2.0, generator=generator)
    random_pred[:, 3:6].uniform_(0.2, 3.0, generator=generator)
    random_pred[:, 6].uniform_(-math.pi, math.pi, generator=generator)
    random_gt = torch.empty((20, 7), device=cpu)
    random_gt.uniform_(-6.0, 6.0, generator=generator)
    random_gt[:, 2].uniform_(-1.0, 2.0, generator=generator)
    random_gt[:, 3:6].uniform_(0.2, 3.0, generator=generator)
    random_gt[:, 6].uniform_(-math.pi, math.pi, generator=generator)
    pred = torch.cat((pred, random_pred), dim=0)
    gt = torch.cat((gt, random_gt), dim=0)
    scores = torch.tensor([
        [0.95, 0.05], [0.90, 0.10], [0.80, 0.20],
        [0.70, 0.85], [0.60, 0.50], [0.55, 0.45],
    ], dtype=torch.float32, device=cpu)
    random_scores = torch.rand((26, 3), generator=generator, device=cpu)
    scores = torch.cat((torch.cat((scores, scores[:, :1]), dim=1), random_scores), dim=0)
    return (pred.unsqueeze(0).to(device), gt.unsqueeze(0).to(device),
            scores.to(device))


def _tensor_digest(value) -> str:
    raw = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _cuda_pairwise(torch, pred, gt):
    from mmcv.ops import diff_iou_rotated_3d
    q, g = pred.shape[1], gt.shape[1]
    pred_pairs = pred[:, :, None].expand(-1, -1, g, -1).reshape(1, q * g, 7)
    gt_pairs = gt[:, None].expand(-1, q, -1, -1).reshape(1, q * g, 7)
    return diff_iou_rotated_3d(pred_pairs, gt_pairs).reshape(q, g)


def _npu_bev_iou(torch, mx_driving, pred, gt):
    op = getattr(mx_driving, 'npu_rotated_iou', None)
    if op is None:
        op = mx_driving.detection.npu_rotated_iou
    pred_bev = torch.stack(
        (pred[..., 0], pred[..., 1], pred[..., 3], pred[..., 4],
         pred[..., 6] * (180.0 / math.pi)), dim=-1)
    gt_bev = torch.stack(
        (gt[..., 0], gt[..., 1], gt[..., 3], gt[..., 4],
         gt[..., 6] * (180.0 / math.pi)), dim=-1)
    return op(pred_bev, gt_bev, False, 0, True, 1e-5, 1e-5)


def _npu_pairwise(torch, mx_driving, pred, gt):
    bev_iou = _npu_bev_iou(torch, mx_driving, pred, gt)
    num_pred, num_gt = pred.shape[1], gt.shape[1]
    # DrivingSDK runs on NPU and its minimum/maximum kernels do not reliably
    # broadcast [B,N,1] with [B,1,M]. Materialize the pairwise shape.
    pred_z_min = (pred[..., 2] - pred[..., 5] / 2).unsqueeze(-1).expand(
        -1, -1, num_gt).contiguous()
    pred_z_max = (pred[..., 2] + pred[..., 5] / 2).unsqueeze(-1).expand(
        -1, -1, num_gt).contiguous()
    gt_z_min = (gt[..., 2] - gt[..., 5] / 2).unsqueeze(-2).expand(
        -1, num_pred, -1).contiguous()
    gt_z_max = (gt[..., 2] + gt[..., 5] / 2).unsqueeze(-2).expand(
        -1, num_pred, -1).contiguous()
    overlap = (torch.minimum(pred_z_max, gt_z_max)
               - torch.maximum(pred_z_min, gt_z_min)).clamp_min(0)
    pred_area = (pred[..., 3] * pred[..., 4]).unsqueeze(-1).expand(
        -1, -1, num_gt).contiguous()
    gt_area = (gt[..., 3] * gt[..., 4]).unsqueeze(-2).expand(
        -1, num_pred, -1).contiguous()
    gt_volume = (gt[..., 3] * gt[..., 4] * gt[..., 5]).unsqueeze(-2).expand(
        -1, num_pred, -1).contiguous()
    pred_volume = (pred[..., 3] * pred[..., 4] * pred[..., 5]).unsqueeze(
        -1).expand(-1, -1, num_gt).contiguous()
    # IoU = intersection / (area_pred + area_gt - intersection). Recover the
    # intersection rather than multiplying IoU by one box's area.
    inter_bev = bev_iou * (pred_area + gt_area) / (1 + bev_iou).clamp_min(1e-8)
    intersection = inter_bev * overlap
    return intersection / (pred_volume + gt_volume - intersection).clamp_min(1e-8)


def _cuda_matched(torch, pred, gt):
    from mmcv.ops import diff_iou_rotated_3d
    return diff_iou_rotated_3d(pred, gt).squeeze(0)


def _npu_matched(torch, mx_driving, pred, gt):
    op = getattr(mx_driving, 'diff_iou_rotated_2d', None)
    if op is None:
        op = mx_driving.detection.diff_iou_rotated_2d
    pred_bev = torch.stack(
        (pred[..., 0], pred[..., 1], pred[..., 3], pred[..., 4], pred[..., 6]),
        dim=-1)
    gt_bev = torch.stack(
        (gt[..., 0], gt[..., 1], gt[..., 3], gt[..., 4], gt[..., 6]), dim=-1)
    bev_iou = op(pred_bev, gt_bev)
    z_overlap = (torch.minimum(pred[..., 2] + pred[..., 5] / 2,
                               gt[..., 2] + gt[..., 5] / 2)
                 - torch.maximum(pred[..., 2] - pred[..., 5] / 2,
                                 gt[..., 2] - gt[..., 5] / 2)).clamp_min(0)
    pred_area = pred[..., 3] * pred[..., 4]
    gt_area = gt[..., 3] * gt[..., 4]
    inter_bev = bev_iou * (pred_area + gt_area) / (1 + bev_iou).clamp_min(1e-8)
    intersection = inter_bev * z_overlap
    volume_pred = pred[..., 3] * pred[..., 4] * pred[..., 5]
    volume_gt = gt[..., 3] * gt[..., 4] * gt[..., 5]
    return intersection / (volume_pred + volume_gt - intersection).clamp_min(1e-8)


def _cuda_nms(torch, boxes, scores, threshold):
    from mmcv.ops import nms3d
    return {
        str(cls): nms3d(boxes, scores[:, cls], threshold).detach().cpu().tolist()
        for cls in range(scores.shape[1])
    }


def _npu_nms(torch, mx_driving, boxes, scores, threshold):
    op = getattr(mx_driving, 'nms3d', None)
    if op is None:
        op = mx_driving.detection.nms3d
    return {
        str(cls): op(boxes, scores[:, cls], threshold).detach().cpu().tolist()
        for cls in range(scores.shape[1])
    }


def _device_info(torch, device):
    info = {'device': str(device), 'torch': torch.__version__}
    if device.type == 'cuda':
        info['device_name'] = torch.cuda.get_device_name(device)
    else:
        import torch_npu
        info['torch_npu'] = getattr(torch_npu, '__version__', 'installed')
        info['device_name'] = torch.npu.get_device_name(0)
    return info


def run_benchmark(device_name: str, warmup: int, repeats: int) -> Dict[str, object]:
    torch = importlib.import_module('torch')
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        device = torch.device('cuda')
        mx_driving = None
    elif device_name == 'npu':
        importlib.import_module('torch_npu')
        if not torch.npu.is_available():
            raise RuntimeError('NPU is unavailable')
        # Match the production import order. MMCV/mmdet3d may load CANN
        # operator libraries before mx_driving; an isolated mx_driving import
        # can otherwise hide an in-process libopapi.so conflict.
        importlib.import_module('recondet.recondet_head')
        device = torch.device('npu')
        mx_driving = importlib.import_module('mx_driving')
    else:
        raise ValueError('device must be cuda or npu')

    pred, gt, scores = _build_inputs(torch, device)
    if device_name == 'cuda':
        pairwise_call = lambda: _cuda_pairwise(torch, pred, gt)
        matched_call = lambda: _cuda_matched(torch, pred[:, :gt.shape[1]], gt)
        nms_call = lambda: _cuda_nms(torch, pred[0], scores, 0.25)
    else:
        pairwise_call = lambda: _npu_pairwise(torch, mx_driving, pred, gt)
        matched_call = lambda: _npu_matched(
            torch, mx_driving, pred[:, :gt.shape[1]], gt)
        nms_call = lambda: _npu_nms(torch, mx_driving, pred[0], scores, 0.25)

    pairwise, pairwise_timing = _timed(
        torch, device, pairwise_call, warmup, repeats)
    matched, matched_timing = _timed(
        torch, device, matched_call, warmup, repeats)
    pred_grad = pred[:, :gt.shape[1]].detach().clone().requires_grad_(True)
    if device_name == 'cuda':
        backward_call = lambda: _cuda_matched(torch, pred_grad, gt).sum()
    else:
        backward_call = lambda: _npu_matched(
            torch, mx_driving, pred_grad, gt).sum()
    for _ in range(warmup):
        pred_grad.grad = None
        backward_call().backward()
    _sync(torch, device)
    backward_times = []
    gradient = None
    for _ in range(repeats):
        pred_grad.grad = None
        _sync(torch, device)
        start = time.perf_counter()
        backward_call().backward()
        _sync(torch, device)
        backward_times.append((time.perf_counter() - start) * 1000.0)
        gradient = pred_grad.grad.detach().clone()
    nms, nms_timing = _timed(torch, device, nms_call, warmup, repeats)
    del nms_timing['repeats'], nms_timing['warmup']
    return {
        'status': 'ok',
        'device': _device_info(torch, device),
        'config': {'warmup': warmup, 'repeats': repeats, 'seed': 20260913},
        'inputs': {
            'pred_shape': list(pred.shape),
            'gt_shape': list(gt.shape),
            'scores_shape': list(scores.shape),
            'pred_sha256': _tensor_digest(pred),
            'gt_sha256': _tensor_digest(gt),
            'scores_sha256': _tensor_digest(scores),
        },
        'results': {
            'pairwise_3d_iou': {
                **summarize_array(pairwise), 'timing': pairwise_timing},
            'matched_3d_iou_backward': {
                **summarize_array(matched),
                'gradient': summarize_array(gradient),
                'timing': {
                    **matched_timing,
                    'backward_mean_ms': round(statistics.mean(backward_times), 4),
                    'backward_median_ms': round(statistics.median(backward_times), 4),
                }},
            'classwise_nms': {'indices': nms, 'timing': nms_timing},
        },
    }


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'npu'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', nargs=2, type=Path, metavar=('CUDA_JSON', 'NPU_JSON'))
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args(argv)
    if args.dry_run:
        payload = {'status': 'ok', 'mode': 'dry-run', 'operations': OPERATIONS}
    elif args.compare:
        with args.compare[0].open() as handle:
            cuda = json.load(handle)
        with args.compare[1].open() as handle:
            npu = json.load(handle)
        payload = compare_payloads(cuda, npu, args.atol, args.rtol)
    elif args.device:
        try:
            payload = run_benchmark(args.device, args.warmup, args.repeats)
        except Exception as exc:
            payload = {
                'status': 'failed',
                'device': args.device,
                'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(),
            }
    else:
        parser.error('provide --device, --compare, or --dry-run')
        return 2

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output and not args.compare and not args.dry_run:
        args.output.write_text(text + '\n', encoding='utf-8')
    if args.json or args.dry_run or args.compare:
        print(text)
    else:
        print(text)
    # A real device benchmark must fail at the shell level when an operator
    # raises. Previously the exception was serialized but the process exited
    # zero unless --strict was supplied, hiding broken NPU runs in scripts.
    if payload.get('status') != 'ok' and (args.device or args.strict):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
