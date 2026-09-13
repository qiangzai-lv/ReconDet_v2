"""Device-dispatched geometry operators used by ReconDet.

The CUDA path remains MMCV's fused implementation. Ascend uses the official
DrivingSDK 2D rotated-IoU and 3D NMS operators, with differentiable PyTorch
math to restore the 3D IoU from BEV IoU and vertical overlap.
"""

import importlib

import torch


def _driving_op(name):
    driving = importlib.import_module('mx_driving')
    op = getattr(driving, name, None)
    if op is not None:
        return op
    detection = getattr(driving, 'detection', None)
    op = getattr(detection, name, None) if detection is not None else None
    if op is None:
        raise AttributeError(f'mx_driving does not provide {name}')
    return op


def _validate_aligned_boxes(boxes_a, boxes_b):
    if boxes_a.ndim != 3 or boxes_b.ndim != 3:
        raise ValueError('aligned boxes must have shape [B, N, 7]')
    if boxes_a.shape != boxes_b.shape or boxes_a.shape[-1] != 7:
        raise ValueError(
            'aligned boxes must have identical [B, N, 7] shapes')


def npu_aligned_3d_iou(boxes_a, boxes_b):
    """Compute aligned differentiable 3D IoU with DrivingSDK operators.

    Args:
        boxes_a, boxes_b: Float tensors with shape ``[B, N, 7]`` and box
            format ``[x, y, z, w, l, h, yaw]``. Yaw is in radians.
    """
    _validate_aligned_boxes(boxes_a, boxes_b)
    op = _driving_op('diff_iou_rotated_2d')
    # DrivingSDK's differentiable IoU contract is float32. Casting here keeps
    # the wrapper usable under AMP while preserving gradients through cast().
    boxes_a = boxes_a.float()
    boxes_b = boxes_b.float()
    bev_a = torch.stack(
        (boxes_a[..., 0], boxes_a[..., 1], boxes_a[..., 3],
         boxes_a[..., 4], boxes_a[..., 6]), dim=-1).contiguous()
    bev_b = torch.stack(
        (boxes_b[..., 0], boxes_b[..., 1], boxes_b[..., 3],
         boxes_b[..., 4], boxes_b[..., 6]), dim=-1).contiguous()
    bev_iou = op(bev_a, bev_b)

    area_a = boxes_a[..., 3] * boxes_a[..., 4]
    area_b = boxes_b[..., 3] * boxes_b[..., 4]
    bev_intersection = bev_iou * (area_a + area_b) / (
        1.0 + bev_iou).clamp_min(1e-8)
    z_overlap = (
        torch.minimum(boxes_a[..., 2] + boxes_a[..., 5] / 2,
                      boxes_b[..., 2] + boxes_b[..., 5] / 2)
        - torch.maximum(boxes_a[..., 2] - boxes_a[..., 5] / 2,
                        boxes_b[..., 2] - boxes_b[..., 5] / 2)
    ).clamp_min(0)
    intersection = bev_intersection * z_overlap
    volume_a = area_a * boxes_a[..., 5]
    volume_b = area_b * boxes_b[..., 5]
    return intersection / (volume_a + volume_b - intersection).clamp_min(1e-8)


def diff_iou_rotated_3d(boxes_a, boxes_b):
    """Dispatch ReconDet's aligned 3D IoU to CUDA or Ascend."""
    if boxes_a.device.type == 'npu':
        return npu_aligned_3d_iou(boxes_a, boxes_b)
    from mmcv.ops import diff_iou_rotated_3d as mmcv_diff_iou_rotated_3d
    return mmcv_diff_iou_rotated_3d(boxes_a, boxes_b)


def npu_nms3d(boxes, scores, iou_threshold):
    """Call the official Ascend 3D NMS operator."""
    return _driving_op('nms3d')(boxes, scores, iou_threshold)


def nms3d(boxes, scores, iou_threshold):
    """Dispatch 3D NMS to CUDA/MMCV or Ascend/DrivingSDK."""
    if boxes.device.type == 'npu':
        return npu_nms3d(boxes, scores, iou_threshold)
    from mmcv.ops import nms3d as mmcv_nms3d
    return mmcv_nms3d(boxes, scores, iou_threshold)
