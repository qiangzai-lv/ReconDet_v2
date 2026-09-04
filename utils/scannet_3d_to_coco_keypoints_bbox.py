"""Export ScanNet views with keypoints and point-cloud-derived 2D boxes."""

import argparse
import json
import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from scannet_3d_to_coco_keypoints import (
    FACE_NORMALS, _load_points, _project, _sample_indices, _surface_points)

LOGGER = logging.getLogger('scannet_3d_to_coco_keypoints_bbox')


def _farthest_sample(points, center, max_points):
    if len(points) <= max_points:
        return points
    selected = np.empty(max_points, dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float32)
    selected[0] = np.linalg.norm(points - center[None], axis=1).argmax()
    for index in range(1, max_points):
        distance = np.sum((points - points[selected[index - 1]]) ** 2, axis=1)
        distances = np.minimum(distances, distance)
        distances[selected[:index]] = -1
        selected[index] = distances.argmax()
    return points[selected]


def convert(args):
    root = Path(args.data_root).resolve()
    ann_path = Path(args.ann_file)
    if not ann_path.is_absolute():
        ann_path = root / ann_path
    with ann_path.open('rb') as handle:
        payload = pickle.load(handle)
    data_list = payload['data_list']
    if args.max_scenes > 0:
        data_list = data_list[:args.max_scenes]
    categories = sorted(payload['metainfo']['categories'].items(), key=lambda x: x[1])
    rng = np.random.default_rng(args.seed)
    coco = {
        'info': {'description': 'ScanNet reconstruction keypoints and point-cloud boxes'},
        'licenses': [], 'images': [], 'annotations': [],
        'categories': [{'id': i + 1, 'name': n, 'supercategory': 'object'}
                       for n, i in categories],
    }
    stats = Counter()
    image_id = annotation_id = 1
    for scene_index, info in enumerate(data_list):
        paths = [Path(p) for p in info['img_paths']]
        indices = _sample_indices(len(paths), args.num_views, args.sampling, rng)
        axis = np.asarray(info['axis_align_matrix'], dtype=np.float32)
        points = _load_points(root, info, axis)
        objects = []
        for obj in info.get('instances', []):
            box = np.asarray(obj['bbox_3d'][:6], dtype=np.float32)
            lower, upper = box[:3] - box[3:6] / 2., box[:3] + box[3:6] / 2.
            inside = np.all((points >= lower) & (points <= upper), axis=1)
            object_points = _farthest_sample(points[inside], box[:3], args.max_points)
            face_points = _surface_points(points, box[:3], box[3:6])
            objects.append((box, object_points, face_points,
                            int(obj['bbox_label_3d'])))
        scene_id = str(info.get('scene_id') or paths[0].parent.name)
        for index in indices:
            index = int(index)
            source = paths[index]
            source_abs = source if source.is_absolute() else root / source
            with Image.open(source_abs) as image:
                width, height = image.size
            pose = np.asarray(info['lidar2cam'][index], dtype=np.float32)
            extrinsic = np.linalg.inv(axis @ pose)
            camera_center = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
            image_record = {'id': image_id, 'file_name': str(source),
                            'width': width, 'height': height,
                            'scene_id': scene_id, 'view_index': index}
            coco['images'].append(image_record)
            for box, object_points, face_points, label in objects:
                object_center = box[:3]
                face_scores = (FACE_NORMALS * (camera_center - object_center)).sum(axis=1)
                face_2d, face_depth = _project(face_points, extrinsic,
                                                np.asarray(info['cam2img']), width, height)
                face_visible = (face_depth > 1e-5)
                face_visible &= np.isfinite(face_2d).all(axis=1)
                face_visible &= (face_2d[:, 0] >= 0) & (face_2d[:, 0] < width)
                face_visible &= (face_2d[:, 1] >= 0) & (face_2d[:, 1] < height)
                selected = np.flatnonzero(face_visible)
                selected = selected[np.argsort(-face_scores[selected])]
                if len(selected) < 3:
                    fallback = np.argsort(-face_scores)
                    selected = np.concatenate([selected,
                                               fallback[~np.isin(fallback, selected)]])
                selected = selected[:3]
                key3d = np.concatenate([object_center[None], face_points[selected]], axis=0)
                key2d, key_depth = _project(key3d, extrinsic,
                                            np.asarray(info['cam2img']), width, height)
                center_visible = (key_depth[0] > 1e-5 and
                                  np.isfinite(key2d[0]).all() and
                                  0 <= key2d[0, 0] < width and
                                  0 <= key2d[0, 1] < height)
                if not center_visible:
                    stats['center_invisible'] += 1
                    continue
                visibility = np.ones(4, dtype=np.int64)
                face_visible = (key_depth[1:] > 1e-5)
                face_visible &= np.isfinite(key2d[1:]).all(axis=1)
                face_visible &= (key2d[1:, 0] >= 0) & (key2d[1:, 0] < width)
                face_visible &= (key2d[1:, 1] >= 0) & (key2d[1:, 1] < height)
                visibility[1:] = face_visible.astype(np.int64)

                projected_points, point_depth = _project(
                    object_points, extrinsic, np.asarray(info['cam2img']), width, height)
                point_valid = point_depth > 1e-5
                point_valid &= np.isfinite(projected_points).all(axis=1)
                point_valid &= (projected_points[:, 0] >= 0) & (projected_points[:, 0] < width)
                point_valid &= (projected_points[:, 1] >= 0) & (projected_points[:, 1] < height)
                if not point_valid.any():
                    stats['no_visible_surface_points'] += 1
                    continue
                visible_pixels = projected_points[point_valid]
                x1, y1 = visible_pixels.min(axis=0)
                x2, y2 = visible_pixels.max(axis=0)
                if x2 <= x1 or y2 <= y1:
                    stats['invalid_bbox'] += 1
                    continue
                flat = [value for point, visible in zip(key2d, visibility)
                        for value in (float(point[0]), float(point[1]), int(visible))]
                coco['annotations'].append({
                    'id': annotation_id, 'image_id': image_id,
                    'category_id': label + 1,
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'area': float((x2 - x1) * (y2 - y1)), 'iscrowd': 0,
                    'keypoints': flat,
                    'keypoints_2d': key2d.reshape(-1).astype(float).tolist(),
                    'keypoints_visibility': visibility.tolist(),
                    'keypoints_3d': key3d.reshape(-1).astype(float).tolist(),
                    'keypoint_faces': selected.astype(int).tolist(),
                    'num_keypoints': 4,
                    'surface_points_used': int(len(object_points)),
                })
                annotation_id += 1; stats['annotations'] += 1
            image_id += 1; stats['images'] += 1
        if (scene_index + 1) % args.log_interval == 0 or scene_index + 1 == len(data_list):
            LOGGER.info('Progress %d/%d scenes: %d images, %d annotations',
                        scene_index + 1, len(data_list), stats['images'], stats['annotations'])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, indent=2))
    LOGGER.info('Wrote %d images and %d combined annotations to %s',
                len(coco['images']), len(coco['annotations']), output)
    LOGGER.info('Stats: %s', dict(stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True); parser.add_argument('--ann-file', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--num-views', required=True, type=int)
    parser.add_argument('--max-points', type=int, default=50)
    parser.add_argument('--sampling', choices=('random', 'uniform'), default='uniform')
    parser.add_argument('--seed', type=int, default=0); parser.add_argument('--max-scenes', type=int, default=-1)
    parser.add_argument('--log-interval', type=int, default=50)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    if args.max_points <= 0 or args.log_interval <= 0:
        raise ValueError('--max-points and --log-interval must be positive')
    convert(args)


if __name__ == '__main__':
    main()
