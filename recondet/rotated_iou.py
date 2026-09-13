"""Backend-portable aligned and pairwise oriented 3D IoU helpers."""

import torch


def _validate_boxes(name, boxes):
    if not isinstance(boxes, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor')
    if boxes.ndim < 2 or boxes.shape[-1] != 7:
        raise ValueError(f'{name} must have shape [..., 7]')
    if not torch.isfinite(boxes).all():
        raise ValueError(f'{name} contains non-finite values')
    if (boxes[..., 3:6] <= 0).any():
        raise ValueError(f'{name} sizes must be positive')


def _bev_boxes(boxes):
    return boxes[..., (0, 1, 3, 4, 6)]


def combine_bev_iou_with_height(boxes_a, boxes_b, bev_iou, eps=1e-6):
    """Combine aligned BEV rotated IoU with vertical overlap into 3D IoU."""
    _validate_boxes('boxes_a', boxes_a)
    _validate_boxes('boxes_b', boxes_b)
    if boxes_a.shape != boxes_b.shape:
        raise ValueError('aligned boxes must have identical shapes')
    expected_shape = boxes_a.shape[:-1]
    if bev_iou.shape != expected_shape:
        raise ValueError(
            f'bev_iou must have shape {expected_shape}, got {bev_iou.shape}')

    bev_iou = bev_iou.to(dtype=boxes_a.dtype, device=boxes_a.device)
    area_a = boxes_a[..., 3] * boxes_a[..., 4]
    area_b = boxes_b[..., 3] * boxes_b[..., 4]
    bev_intersection = bev_iou.clamp(0, 1) * (area_a + area_b) / (
        1 + bev_iou.clamp(0, 1)).clamp_min(eps)
    z_a_min = boxes_a[..., 2] - boxes_a[..., 5] * 0.5
    z_a_max = boxes_a[..., 2] + boxes_a[..., 5] * 0.5
    z_b_min = boxes_b[..., 2] - boxes_b[..., 5] * 0.5
    z_b_max = boxes_b[..., 2] + boxes_b[..., 5] * 0.5
    z_overlap = (torch.minimum(z_a_max, z_b_max)
                  - torch.maximum(z_a_min, z_b_min)).clamp_min(0)
    intersection = bev_intersection * z_overlap
    volume_a = area_a * boxes_a[..., 5]
    volume_b = area_b * boxes_b[..., 5]
    union = (volume_a + volume_b - intersection).clamp_min(eps)
    return (intersection / union).clamp(0, 1)


def _driving_bev_iou(boxes_a, boxes_b):
    import mx_driving

    operator = getattr(mx_driving, 'diff_iou_rotated_2d', None)
    if operator is None:
        raise RuntimeError('mx_driving.diff_iou_rotated_2d is unavailable')
    return operator(boxes_a, boxes_b)


def _mmcv_aligned_iou(boxes_a, boxes_b):
    from mmcv.ops import diff_iou_rotated_3d

    return diff_iou_rotated_3d(boxes_a, boxes_b)


def _resolve_backend(boxes, backend):
    if backend not in ('auto', 'driving', 'mmcv'):
        raise ValueError("backend must be 'auto', 'driving', or 'mmcv'")
    if backend == 'auto':
        return 'driving' if boxes.device.type == 'npu' else 'mmcv'
    return backend


def rotated_iou_3d_aligned(boxes_a, boxes_b, backend='auto'):
    """Compute aligned oriented 3D IoU with shape ``[..., 7]``."""
    _validate_boxes('boxes_a', boxes_a)
    _validate_boxes('boxes_b', boxes_b)
    if boxes_a.shape != boxes_b.shape:
        raise ValueError('aligned boxes must have identical shapes')
    selected_backend = _resolve_backend(boxes_a, backend)
    if selected_backend == 'mmcv':
        return _mmcv_aligned_iou(boxes_a, boxes_b)
    bev_iou = _driving_bev_iou(_bev_boxes(boxes_a), _bev_boxes(boxes_b))
    return combine_bev_iou_with_height(boxes_a, boxes_b, bev_iou)


def rotated_iou_3d_pairwise(boxes_a, boxes_b, backend='auto', chunk_size=1024):
    """Compute pairwise oriented 3D IoU: ``[N,7] x [M,7] -> [N,M]``."""
    _validate_boxes('boxes_a', boxes_a)
    _validate_boxes('boxes_b', boxes_b)
    if boxes_a.ndim != 2 or boxes_b.ndim != 2:
        raise ValueError('pairwise boxes must have shape [N, 7] and [M, 7]')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if boxes_a.device != boxes_b.device:
        raise ValueError('pairwise boxes must be on the same device')
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
    selected_backend = _resolve_backend(boxes_a, backend)
    if selected_backend == 'mmcv':
        pred = boxes_a[:, None].expand(-1, boxes_b.shape[0], -1).reshape(
            1, -1, 7)
        target = boxes_b[None].expand(boxes_a.shape[0], -1, -1).reshape(
            1, -1, 7)
        return _mmcv_aligned_iou(pred, target).reshape(
            boxes_a.shape[0], boxes_b.shape[0])

    outputs = []
    num_gt = boxes_b.shape[0]
    for start in range(0, boxes_a.shape[0], chunk_size):
        pred_chunk = boxes_a[start:start + chunk_size]
        num_pred = pred_chunk.shape[0]
        pred = pred_chunk[:, None].expand(-1, num_gt, -1).reshape(-1, 7)
        target = boxes_b[None].expand(num_pred, -1, -1).reshape(-1, 7)
        iou = rotated_iou_3d_aligned(
            pred.unsqueeze(0), target.unsqueeze(0), backend='driving')
        outputs.append(iou.reshape(num_pred, num_gt))
    return torch.cat(outputs, dim=0)


def diff_iou_rotated_3d(boxes_a, boxes_b, backend='auto'):
    """Compatibility-named aligned 3D IoU adapter."""
    return rotated_iou_3d_aligned(boxes_a, boxes_b, backend=backend)
