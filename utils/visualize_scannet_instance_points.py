"""Visualize ScanNet 3D boxes and the points assigned to each instance.

The output is an ASCII PLY containing colored point vertices and colored box
edges. Points inside exactly one box use that instance's color, points inside
multiple boxes are magenta, and background points are gray.
"""

from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import pickle
from typing import Iterable, Tuple

import numpy as np


PALETTE = np.asarray([
    (230, 57, 70), (29, 130, 196), (67, 170, 139), (244, 162, 97),
    (138, 201, 38), (106, 76, 147), (249, 199, 79), (255, 89, 94),
    (39, 125, 161), (144, 190, 109), (233, 196, 106), (25, 130, 196),
], dtype=np.uint8)
BACKGROUND_COLOR = np.asarray((135, 135, 135), dtype=np.uint8)
OVERLAP_COLOR = np.asarray((255, 0, 255), dtype=np.uint8)


def classify_point_membership(
        points: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return point-box membership and owner (-1 background, -2 overlap)."""
    points = np.asarray(points, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'points must have shape [N, 3], got {points.shape}')
    if boxes.ndim != 2 or boxes.shape[1] < 6:
        raise ValueError(f'boxes must have shape [M, >=6], got {boxes.shape}')
    if not len(boxes):
        return (np.zeros((len(points), 0), dtype=bool),
                np.full(len(points), -1, dtype=np.int32))
    centers = boxes[:, :3]
    half_sizes = boxes[:, 3:6] / 2.0
    membership = np.all(
        (points[:, None, :] >= centers[None] - half_sizes[None])
        & (points[:, None, :] <= centers[None] + half_sizes[None]), axis=-1)
    counts = membership.sum(axis=1)
    owner = np.full(len(points), -1, dtype=np.int32)
    single = counts == 1
    owner[single] = membership[single].argmax(axis=1).astype(np.int32)
    owner[counts > 1] = -2
    return membership, owner


def _resolve_point_path(root: Path, info: dict) -> Path:
    lidar = info.get('aligned_pts_path')
    if lidar is None:
        lidar = info.get('lidar_points', {}).get('lidar_path')
    if lidar is None:
        scene_id = str(info.get('scene_id', Path(info['img_paths'][0]).parent.name))
        lidar = Path('points') / f'{scene_id}.bin'
    relative = Path(lidar)
    candidates = (relative, root / relative, root / 'points' / relative.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('point cloud not found; checked: ' +
                            ', '.join(map(str, candidates)))


def load_scene_points(root: Path, info: dict) -> np.ndarray:
    metadata = info.get('lidar_points', {})
    dimensions = int(metadata.get('num_pts_feats', info.get('num_pts_feats', 6)))
    path = _resolve_point_path(root, info)
    values = np.fromfile(path, dtype=np.float32)
    if dimensions < 3 or values.size % dimensions:
        raise ValueError(f'invalid {dimensions}-D point file: {path}')
    points = values.reshape(-1, dimensions)[:, :3]
    if info.get('aligned_pts_path') is None:
        axis = np.asarray(info['axis_align_matrix'], dtype=np.float32)
        if axis.shape != (4, 4):
            raise ValueError(f'axis_align_matrix must be 4x4, got {axis.shape}')
        points = points @ axis[:3, :3].T + axis[:3, 3]
    return points


def _box_corners(box: np.ndarray) -> np.ndarray:
    center = box[:3]
    half = box[3:6] / 2.0
    signs = np.asarray([
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)], dtype=np.float32)
    return center + signs * half


BOX_EDGES = tuple((a, b) for a in range(8) for b in range(a + 1, 8)
                  if bin(a ^ b).count('1') == 1)


def _subsample_indices(owner: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(owner) <= max_points:
        return np.arange(len(owner))
    keep = []
    for group in np.unique(owner):
        ids = np.flatnonzero(owner == group)
        count = max(1, round(max_points * len(ids) / len(owner)))
        keep.append(ids[np.linspace(0, len(ids) - 1, min(count, len(ids)),
                                   dtype=int)])
    selected = np.unique(np.concatenate(keep))
    if len(selected) > max_points:
        selected = selected[np.linspace(0, len(selected) - 1, max_points,
                                        dtype=int)]
    return selected


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray,
               boxes: np.ndarray, box_colors: np.ndarray) -> None:
    corners = np.concatenate([_box_corners(box) for box in boxes], axis=0) \
        if len(boxes) else np.empty((0, 3), dtype=np.float32)
    vertices = np.concatenate([points, corners], axis=0)
    vertex_colors = np.concatenate([
        colors,
        np.repeat(box_colors, 8, axis=0) if len(boxes)
        else np.empty((0, 3), dtype=np.uint8)
    ], axis=0)
    edges = []
    edge_colors = []
    for box_id in range(len(boxes)):
        offset = len(points) + box_id * 8
        for first, second in BOX_EDGES:
            edges.append((offset + first, offset + second))
            edge_colors.append(box_colors[box_id])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='ascii') as handle:
        handle.write('ply\nformat ascii 1.0\n')
        handle.write(f'comment colored ScanNet instance point visualization\n')
        handle.write(f'element vertex {len(vertices)}\n')
        handle.write('property float x\nproperty float y\nproperty float z\n')
        handle.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        handle.write(f'element edge {len(edges)}\n')
        handle.write('property int vertex1\nproperty int vertex2\n')
        handle.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        handle.write('end_header\n')
        for point, color in zip(vertices, vertex_colors):
            handle.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} '
                         f'{int(color[0])} {int(color[1])} {int(color[2])}\n')
        for (first, second), color in zip(edges, edge_colors):
            handle.write(f'{first} {second} {int(color[0])} '
                         f'{int(color[1])} {int(color[2])}\n')


def visualize_scene(root: Path, info: dict, output: Path,
                    max_points: int) -> None:
    points = load_scene_points(root, info)
    instances = info.get('instances', [])
    boxes = np.asarray([item['bbox_3d'][:6] for item in instances],
                       dtype=np.float32)
    if not len(boxes):
        boxes = np.empty((0, 6), dtype=np.float32)
    membership, owner = classify_point_membership(points, boxes)
    indices = _subsample_indices(owner, max_points)
    selected_owner = owner[indices]
    colors = np.repeat(BACKGROUND_COLOR[None], len(indices), axis=0)
    for instance_id in range(len(boxes)):
        colors[selected_owner == instance_id] = PALETTE[instance_id % len(PALETTE)]
    colors[selected_owner == -2] = OVERLAP_COLOR
    box_colors = np.asarray(
        [PALETTE[index % len(PALETTE)] for index in range(len(boxes))],
        dtype=np.uint8)
    _write_ply(output, points[indices], colors, boxes, box_colors)
    scene_id = str(info.get('scene_id', output.stem))
    print(f'{scene_id}: points={len(points)} written={len(indices)} '
          f'instances={len(boxes)} overlap_points={(owner == -2).sum()}')
    for index, instance in enumerate(instances):
        inside = membership[:, index]
        exclusive = inside & (owner == index)
        print(f'  instance={int(instance.get("instance_id", index))} '
              f'label={int(instance.get("bbox_label_3d", -1))} '
              f'box_points={inside.sum()} exclusive={exclusive.sum()} '
              f'overlap={(inside & (owner == -2)).sum()}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--ann-file', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--scene-id')
    parser.add_argument('--max-points', type=int, default=200000)
    args = parser.parse_args()
    if args.max_points == 0 or args.max_points < -1:
        raise ValueError('--max-points must be -1 or positive')
    with args.ann_file.open('rb') as handle:
        payload = pickle.load(handle)
    data_list = payload['data_list']
    selected = [item for item in data_list
                if args.scene_id is None or str(item.get('scene_id',
                    Path(item['img_paths'][0]).parent.name)) == args.scene_id]
    if not selected:
        raise ValueError(f'no scene found for --scene-id {args.scene_id!r}')
    for info in selected:
        scene_id = str(info.get('scene_id', Path(info['img_paths'][0]).parent.name))
        output = args.output_dir / f'{scene_id}_instances.ply'
        visualize_scene(args.data_root, info, output, args.max_points)
        print(f'  wrote {output}')


if __name__ == '__main__':
    main()
