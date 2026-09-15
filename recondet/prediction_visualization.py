"""Scene-level prediction visualization as colored PLY plus JSON metadata."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch


_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
          (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))

_DEFAULT_CLASS_COLORS = {
    0: (230, 25, 75), 1: (60, 180, 75), 2: (255, 225, 25),
    3: (0, 130, 200), 4: (245, 130, 48), 5: (145, 30, 180),
    6: (70, 240, 240), 7: (240, 50, 230), 8: (210, 245, 60),
    9: (250, 190, 190), 10: (0, 128, 128), 11: (230, 190, 255),
    12: (170, 110, 40), 13: (255, 250, 200), 14: (128, 0, 0),
    15: (170, 255, 195), 16: (128, 128, 0), 17: (255, 215, 180),
}


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


def _colored_box_overlay(path, gt_boxes, gt_labels, boxes, labels,
                         class_colors, box_line_step):
    parts, colors = [], []
    gt_centers, gt_sizes = _box_center_size(gt_boxes)
    raw_boxes = _tensor(boxes)
    if raw_boxes.size:
        box_centers, box_sizes = _box_center_size(boxes)
        boxes = np.concatenate((box_centers, box_sizes), axis=1)
    else:
        boxes = np.empty((0, 6), np.float32)
    labels = _tensor(labels).reshape(-1).astype(np.int64)
    if len(boxes) != len(labels):
        raise ValueError('boxes and labels must have the same length')
    for center, size, label in zip(gt_centers, gt_sizes, _tensor(gt_labels).reshape(-1)):
        color = class_colors.get(int(label), (150, 150, 150))
        _append_box(parts, colors, center, size, color, box_line_step)
    for center, size, label in zip(boxes[:, :3], boxes[:, 3:6], labels):
        color = class_colors.get(int(label), (150, 150, 150))
        _append_box(parts, colors, center, size, color, box_line_step)
    if not parts:
        parts.append(np.empty((0, 3), np.float32)); colors.append(np.empty((0, 3), np.uint8))
    write_binary_ply(path, np.concatenate(parts), np.concatenate(colors))


def save_scene_reconstruction_visualizations(
        output_dir, scene_id, gt_points, gt_boxes, gt_labels,
        reconstruction_points, reconstruction_labels,
        reconstruction_boxes_before_nms, reconstruction_labels_before_nms,
        reconstruction_boxes_after_nms, reconstruction_labels_after_nms,
        fallback_boxes, final_boxes, final_labels, class_colors=None,
        box_line_step=0.03, max_points=200000):
    """Write five stage-specific PLY overlays for one aligned scene."""
    output_dir = Path(output_dir) / str(scene_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_colors = dict(_DEFAULT_CLASS_COLORS if class_colors is None else class_colors)
    points = _tensor(reconstruction_points).reshape(-1, 3)
    point_labels = _tensor(reconstruction_labels).reshape(-1).astype(np.int64)
    gt = _tensor(gt_points).reshape(-1, 3)[:, :3]
    if len(points) != len(point_labels):
        raise ValueError('reconstruction points and labels must have same length')
    if len(gt) > max_points:
        gt = gt[np.linspace(0, len(gt) - 1, max_points).astype(np.int64)]
    point_parts = [gt]
    point_colors = [np.full_like(gt, (150, 150, 150), dtype=np.uint8)]
    for label in np.unique(point_labels):
        selected = points[point_labels == label]
        if len(selected):
            point_parts.append(selected)
            point_colors.append(np.broadcast_to(
                class_colors.get(int(label), (150, 150, 150)), selected.shape).copy())
    point_path = output_dir / 'reconstruction_points_vs_gt.ply'
    write_binary_ply(point_path, np.concatenate(point_parts), np.concatenate(point_colors))
    paths = {'reconstruction_points': point_path}
    stages = (
        ('boxes_before_nms', reconstruction_boxes_before_nms, reconstruction_labels_before_nms),
        ('boxes_after_nms', reconstruction_boxes_after_nms, reconstruction_labels_after_nms),
        ('fallback_boxes', fallback_boxes, np.full((len(_tensor(fallback_boxes).reshape(-1, 6)),), -1)),
        ('final_detection', final_boxes, final_labels),
    )
    for name, boxes, labels in stages:
        path = output_dir / f'{name}_vs_gt.ply'
        _colored_box_overlay(path, gt_boxes, gt_labels, boxes, labels,
                             class_colors, box_line_step)
        paths[name] = path
    return paths


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    wrapped_tensor = getattr(value, 'tensor', None)
    if isinstance(wrapped_tensor, torch.Tensor):
        return _tensor(wrapped_tensor)
    return np.asarray(value)


def _box_center_size(boxes):
    gravity_center = getattr(boxes, 'gravity_center', None)
    if gravity_center is not None:
        centers = _tensor(gravity_center)
        tensor = _tensor(getattr(boxes, 'tensor', boxes))
    else:
        tensor = _tensor(getattr(boxes, 'tensor', boxes))
        centers = tensor[:, :3]
    if tensor.ndim != 2 or tensor.shape[1] < 6:
        raise ValueError('3D boxes must have shape [N, >=6]')
    if centers.ndim != 2 or centers.shape[0] != tensor.shape[0]:
        raise ValueError('3D box centers must have shape [N, 3]')
    return centers, tensor[:, 3:6]


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
    pred_scores = _tensor(pred_scores).reshape(-1)
    pred_labels = _tensor(pred_labels).reshape(-1)
    pred_keep = np.isfinite(pred_scores) & (pred_scores > float(score_threshold))
    pred_centers, pred_sizes = _box_center_size(pred_boxes)
    if len(pred_centers) != len(pred_keep):
        raise ValueError('pred_boxes and pred_scores must have the same length')
    pred_centers = pred_centers[pred_keep]
    pred_sizes = pred_sizes[pred_keep]
    pred_scores = pred_scores[pred_keep]
    pred_labels = pred_labels[pred_keep]
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
    labels_pred = pred_labels.astype(np.int64).tolist()
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
                       for label, score in zip(labels_pred, pred_scores.tolist())],
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


def save_scene_cluster_visualization(
        output_dir, scene_id, gt_points, reconstruction_points,
        reconstruction_scores, cluster_centers, cluster_sizes,
        score_threshold=0.1, box_line_step=0.03, max_points=200000):
    """Write reconstruction queries and scene-level clusters as a PLY overlay."""
    output_dir = Path(output_dir) / str(scene_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_points = np.asarray(gt_points, dtype=np.float32)
    if gt_points.ndim != 2 or gt_points.shape[1] < 3:
        raise ValueError('gt_points must have shape [N, >=3]')
    gt_points = gt_points[:, :3]
    if len(gt_points) > max_points:
        ids = np.linspace(0, len(gt_points) - 1, max_points).astype(np.int64)
        gt_points = gt_points[ids]

    queries = _tensor(reconstruction_points).reshape(-1, 3)
    scores = _tensor(reconstruction_scores).reshape(-1)
    keep = np.isfinite(queries).all(axis=1) & np.isfinite(scores)
    keep &= scores > float(score_threshold)
    queries = queries[keep]
    scores = scores[keep]
    centers = _tensor(cluster_centers).reshape(-1, 3)
    sizes = _tensor(cluster_sizes).reshape(-1, 3)
    if len(centers) != len(sizes):
        raise ValueError('cluster_centers and cluster_sizes must have same length')
    valid_clusters = np.isfinite(centers).all(axis=1)
    valid_clusters &= np.isfinite(sizes).all(axis=1) & (sizes > 0).all(axis=1)
    centers = centers[valid_clusters]
    sizes = sizes[valid_clusters]

    parts = [gt_points, queries]
    colors = [
        np.full_like(gt_points, (150, 150, 150), dtype=np.uint8),
        np.full_like(queries, (40, 120, 255), dtype=np.uint8),
    ]
    if len(centers):
        parts.append(centers)
        colors.append(np.full_like(centers, (255, 220, 30), dtype=np.uint8))
        for center, size in zip(centers, sizes):
            _append_box(parts, colors, center, size, (255, 220, 30),
                        box_line_step)

    ply_path = output_dir / 'cluster_overlay.ply'
    write_binary_ply(ply_path, np.concatenate(parts), np.concatenate(colors))
    metadata = {
        'scene_id': str(scene_id),
        'coordinate_system': 'axis_aligned_metric',
        'score_threshold': float(score_threshold),
        'colors': {
            'scene_points': 'gray',
            'reconstruction_query': 'blue',
            'cluster_center_and_extent': 'yellow',
        },
        'reconstruction_query_count': int(len(queries)),
        'cluster_count': int(len(centers)),
        'cluster_centers': centers.tolist(),
        'cluster_sizes': sizes.tolist(),
        'ply': str(ply_path),
    }
    json_path = output_dir / 'cluster_overlay.json'
    json_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return ply_path, json_path
