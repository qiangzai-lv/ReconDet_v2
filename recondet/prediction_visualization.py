"""Scene-level prediction visualization as colored PLY plus JSON metadata."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch


_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
          (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))


def _corners(center, size):
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
        dtype=np.float32)
    return np.asarray(center, dtype=np.float32) + signs * (
        np.asarray(size, dtype=np.float32) / 2.0)


def _line(a, b, step=0.03):
    a, b = np.asarray(a), np.asarray(b)
    count = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    return np.linspace(a, b, count, dtype=np.float32)


def _append_box(parts, colors, center, size, color, step):
    corners = _corners(center, size)
    for a, b in _EDGES:
        points = _line(corners[a], corners[b], step)
        parts.append(points)
        colors.append(np.broadcast_to(color, points.shape).copy())


def write_binary_ply(path, points, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype='<f4')
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError('points and colors must both have shape [N, 3]')
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode('ascii')
    with path.open('wb') as handle:
        handle.write(header)
        for point, color in zip(points, colors):
            handle.write(struct.pack('<fffBBB', *point, *color))


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def _box_center_size(boxes):
    tensor = _tensor(getattr(boxes, 'tensor', boxes))
    if tensor.ndim != 2 or tensor.shape[1] < 6:
        raise ValueError('3D boxes must have shape [N, >=6]')
    return tensor[:, :3], tensor[:, 3:6]


def transform_aabb_boxes(boxes, axis_align_matrix):
    """Transform raw axis-aligned boxes and re-AABB them in aligned space."""
    centers, sizes = _box_center_size(boxes)
    matrix = np.asarray(axis_align_matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError('axis_align_matrix must have shape [4, 4]')
    transformed = []
    for center, size in zip(centers, sizes):
        corners = _corners(center, size)
        corners = corners @ matrix[:3, :3].T + matrix[:3, 3]
        minimum = corners.min(axis=0)
        maximum = corners.max(axis=0)
        transformed.append(np.concatenate(((minimum + maximum) / 2,
                                           maximum - minimum)))
    if not transformed:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(transformed, dtype=np.float32)


def save_scene_prediction_visualization(
        output_dir, scene_id, gt_points, gt_boxes, gt_labels,
        pred_boxes, pred_scores, pred_labels, reconstruction_points,
        reconstruction_scores, class_names, score_threshold=0.1,
        box_line_step=0.03, max_points=200000):
    """Write one scene overlay in the axis-aligned metric coordinate system."""
    output_dir = Path(output_dir) / str(scene_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_points = np.asarray(gt_points, dtype=np.float32)
    if gt_points.ndim != 2 or gt_points.shape[1] < 3:
        raise ValueError('gt_points must have shape [N, >=3]')
    gt_points = gt_points[:, :3]
    if len(gt_points) > max_points:
        ids = np.linspace(0, len(gt_points) - 1, max_points).astype(np.int64)
        gt_points = gt_points[ids]
    parts = [gt_points]
    colors = [np.full_like(gt_points, (150, 150, 150), dtype=np.uint8)]
    # Both the point cloud and GT boxes are already in the axis-aligned
    # metric coordinate system.  Applying axis_align_matrix here would
    # transform the GT boxes a second time and misalign them with the cloud.
    gt_centers, gt_sizes = _box_center_size(gt_boxes)
    pred_centers, pred_sizes = _box_center_size(pred_boxes)
    for center, size in zip(gt_centers, gt_sizes):
        _append_box(parts, colors, center, size, (40, 220, 70), box_line_step)
    for center, size in zip(pred_centers, pred_sizes):
        _append_box(parts, colors, center, size, (230, 40, 40), box_line_step)
    recon = _tensor(reconstruction_points).reshape(-1, 3)
    scores = _tensor(reconstruction_scores).reshape(-1)
    keep = np.isfinite(recon).all(axis=1) & np.isfinite(scores)
    keep &= scores > float(score_threshold)
    recon = recon[keep]
    if len(recon):
        parts.append(recon)
        colors.append(np.full_like(recon, (40, 120, 255), dtype=np.uint8))
    ply_path = output_dir / 'prediction_overlay.ply'
    write_binary_ply(ply_path, np.concatenate(parts), np.concatenate(colors))
    labels_gt = _tensor(gt_labels).astype(np.int64).tolist()
    labels_pred = _tensor(pred_labels).astype(np.int64).tolist()
    metadata = {
        'scene_id': str(scene_id),
        'coordinate_system': 'axis_aligned_metric',
        'colors': {'gt_bbox': 'green', 'pred_bbox': 'red', 'reconstruction_query': 'blue'},
        'score_threshold': float(score_threshold),
        'gt_boxes': [{'label': int(label), 'category': class_names[int(label)]
                      if 0 <= int(label) < len(class_names) else 'unknown'}
                     for label in labels_gt],
        'pred_boxes': [{'label': int(label), 'category': class_names[int(label)]
                        if 0 <= int(label) < len(class_names) else 'unknown',
                        'score': float(score)}
                       for label, score in zip(labels_pred, _tensor(pred_scores).tolist())],
        'reconstruction_query_count': int(len(recon)),
        'gt_point_bounds': {
            'min': gt_points.min(axis=0).tolist(),
            'max': gt_points.max(axis=0).tolist(),
        },
        'gt_box_bounds': {
            'min': (gt_centers - gt_sizes / 2).min(axis=0).tolist()
            if len(gt_centers) else [],
            'max': (gt_centers + gt_sizes / 2).max(axis=0).tolist()
            if len(gt_centers) else [],
        },
        'pred_box_bounds': {
            'min': (pred_centers - pred_sizes / 2).min(axis=0).tolist()
            if len(pred_centers) else [],
            'max': (pred_centers + pred_sizes / 2).max(axis=0).tolist()
            if len(pred_centers) else [],
        },
        'ply': str(ply_path),
    }
    json_path = output_dir / 'prediction_overlay.json'
    json_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return ply_path, json_path
