"""Visualize 3D centers recovered from generated ScanNet 2D annotations."""

from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import json
import pickle
from typing import Dict, Iterable, List

import numpy as np

from utils.visualize_scannet_instance_points import (
    BACKGROUND_COLOR, BOX_EDGES, PALETTE, _box_corners, load_scene_points)


def collect_centers_by_view(annotations: Iterable[dict], scene_id: str,
                            image_view_by_id: Dict[int, tuple] = None
                            ) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for annotation in annotations:
        if str(annotation.get('scene_id', '')) != str(scene_id):
            continue
        center = annotation.get('center_3d')
        if center is None:
            continue
        center = np.asarray(center, dtype=np.float32)
        if center.shape != (3,) or not np.isfinite(center).all():
            continue
        view_index = annotation.get('view_index')
        if view_index is None and image_view_by_id is not None:
            image_info = image_view_by_id.get(int(annotation['image_id']))
            if image_info is not None:
                image_scene, view_index = image_info
                if str(image_scene) != str(scene_id):
                    continue
        if view_index is None:
            raise ValueError('center annotation is missing view_index and '
                             'no image mapping was provided')
        item = dict(annotation)
        item['center_3d'] = center.tolist()
        item['view_index'] = int(view_index)
        grouped.setdefault(int(view_index), []).append(item)
    return grouped


def _subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def _write_ply(path: Path, cloud: np.ndarray, cloud_colors: np.ndarray,
               boxes: np.ndarray, box_colors: np.ndarray,
               centers: Iterable[dict]) -> None:
    center_items = list(centers)
    corners = (np.concatenate([_box_corners(box) for box in boxes], axis=0)
               if len(boxes) else np.empty((0, 3), dtype=np.float32))
    center_points = (np.asarray([item['center_3d'] for item in center_items],
                                dtype=np.float32)
                     if center_items else np.empty((0, 3), dtype=np.float32))
    center_colors = np.asarray([
        PALETTE[int(item.get('instance_id_3d', 0)) % len(PALETTE)]
        for item in center_items
    ], dtype=np.uint8) if center_items else np.empty((0, 3), dtype=np.uint8)
    vertices = np.concatenate([cloud, corners, center_points], axis=0)
    colors = np.concatenate([
        cloud_colors,
        np.repeat(box_colors, 8, axis=0) if len(boxes)
        else np.empty((0, 3), dtype=np.uint8),
        center_colors,
    ], axis=0)
    edges = []
    edge_colors = []
    for box_id in range(len(boxes)):
        offset = len(cloud) + box_id * 8
        for first, second in BOX_EDGES:
            edges.append((offset + first, offset + second))
            edge_colors.append(box_colors[box_id])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='ascii') as handle:
        handle.write('ply\nformat ascii 1.0\n')
        handle.write('comment ScanNet point cloud, boxes, and 2D centers\n')
        handle.write(f'element vertex {len(vertices)}\n')
        handle.write('property float x\nproperty float y\nproperty float z\n')
        handle.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        handle.write(f'element edge {len(edges)}\n')
        handle.write('property int vertex1\nproperty int vertex2\n')
        handle.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        handle.write('end_header\n')
        for point, color in zip(vertices, colors):
            handle.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} '
                         f'{int(color[0])} {int(color[1])} {int(color[2])}\n')
        for (first, second), color in zip(edges, edge_colors):
            handle.write(f'{first} {second} {int(color[0])} '
                         f'{int(color[1])} {int(color[2])}\n')


def visualize_scene(root: Path, info: dict, annotations: List[dict],
                    output_dir: Path, max_points: int) -> None:
    scene_id = str(info.get('scene_id', Path(info['img_paths'][0]).parent.name))
    points = _subsample(load_scene_points(root, info), max_points)
    instances = info.get('instances', [])
    boxes = np.asarray([item['bbox_3d'][:6] for item in instances], dtype=np.float32)
    if not len(boxes):
        boxes = np.empty((0, 6), dtype=np.float32)
    cloud_colors = np.repeat(BACKGROUND_COLOR[None], len(points), axis=0)
    box_colors = np.asarray([
        PALETTE[int(item.get('instance_id', index)) % len(PALETTE)]
        for index, item in enumerate(instances)
    ], dtype=np.uint8)
    centers_by_view = collect_centers_by_view(
        annotations, scene_id, image_view_by_id=info.get('_image_view_by_id'))
    all_centers = [item for view in sorted(centers_by_view)
                   for item in centers_by_view[view]]
    _write_ply(output_dir / f'{scene_id}_all_centers.ply', points,
               cloud_colors, boxes, box_colors, all_centers)
    for view_index, centers in sorted(centers_by_view.items()):
        _write_ply(output_dir / f'{scene_id}_view_{view_index:06d}_centers.ply',
                   points, cloud_colors, boxes, box_colors, centers)
    print(f'{scene_id}: centers={len(all_centers)} views={len(centers_by_view)} '
          f'outputs={1 + len(centers_by_view)}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--ann-file', required=True, type=Path)
    parser.add_argument('--coco', required=True, type=Path,
                        help='Merged keypoints_bbox_train.json or val JSON')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--scene-id')
    parser.add_argument('--max-points', type=int, default=200000)
    args = parser.parse_args()
    if args.max_points == 0 or args.max_points < -1:
        raise ValueError('--max-points must be -1 or positive')
    with args.ann_file.open('rb') as handle:
        payload = pickle.load(handle)
    with args.coco.open('r', encoding='utf-8') as handle:
        coco = json.load(handle)
    annotations_by_scene = {}
    image_view_by_id = {
        int(image['id']): (str(image.get('scene_id', '')),
                           int(image['view_index']))
        for image in coco.get('images', [])
        if 'id' in image and 'view_index' in image
    }
    for annotation in coco.get('annotations', []):
        annotations_by_scene.setdefault(str(annotation.get('scene_id', '')),
                                         []).append(annotation)
    selected = [info for info in payload['data_list']
                if args.scene_id is None or str(info.get('scene_id',
                    Path(info['img_paths'][0]).parent.name)) == args.scene_id]
    if not selected:
        raise ValueError(f'no scene found for --scene-id {args.scene_id!r}')
    for info in selected:
        scene_id = str(info.get('scene_id', Path(info['img_paths'][0]).parent.name))
        info = dict(info)
        info['_image_view_by_id'] = image_view_by_id
        visualize_scene(args.data_root, info, annotations_by_scene.get(scene_id, []),
                        args.output_dir, args.max_points)


if __name__ == '__main__':
    main()
