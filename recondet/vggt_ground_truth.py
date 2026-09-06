from pathlib import Path

import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS


def as_homogeneous(matrix, name='matrix'):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.concatenate(
            [matrix, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    if matrix.shape != (4, 4):
        raise ValueError(f'{name} must have shape [3, 4] or [4, 4]')
    if not np.isfinite(matrix).all():
        raise ValueError(f'{name} contains non-finite values')
    return matrix


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape [N, 3]')
    transform = as_homogeneous(transform, 'transform')
    return points @ transform[:3, :3].T + transform[:3, 3]


def mean_point_distance(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('scale points must have shape [N, 3]')
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError('scale point cloud has no finite points')
    scale = float(np.linalg.norm(points, axis=1).mean())
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f'invalid VGGT GT scale: {scale}')
    return scale


def normalize_aligned_geometry_to_first_camera(
        points_aligned, selected_c2w_aligned,
        scale_points_aligned=None):
    """Normalize aligned ScanNet geometry like VGGT training targets."""
    c2w = np.asarray(selected_c2w_aligned, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or len(c2w) == 0:
        raise ValueError('selected_c2w_aligned must have shape [V, 4, 4]')
    if not np.isfinite(c2w).all():
        raise ValueError('selected_c2w_aligned contains non-finite values')

    first_w2c = np.linalg.inv(c2w[0])
    scale_source = (points_aligned if scale_points_aligned is None
                    else scale_points_aligned)
    scale_points_first = transform_points(scale_source, first_w2c)
    scale = mean_point_distance(scale_points_first)
    normalized_points = transform_points(points_aligned, first_w2c) / scale

    normalized_extrinsics = []
    for camera_to_world in c2w:
        relative_w2c = np.linalg.inv(camera_to_world) @ c2w[0]
        relative_w2c[:3, 3] /= scale
        normalized_extrinsics.append(relative_w2c[:3])
    return (
        normalized_points.astype(np.float32),
        np.asarray(normalized_extrinsics, dtype=np.float32),
        np.float32(scale),
    )


def load_and_resize_depth(path, output_shape, depth_scale=1000.0):
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError('depth_scale must be finite and positive')
    height, width = map(int, output_shape)
    if height <= 0 or width <= 0:
        raise ValueError('output_shape must contain positive height and width')
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f'Could not read depth map: {path}')
    if depth.ndim != 2:
        raise ValueError(f'Depth map must be single-channel: {path}')
    if depth.shape != (height, width):
        depth = cv2.resize(
            depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return depth.astype(np.float32) / float(depth_scale)


def build_vggt_ground_truth(raw_points, axis_align_matrix, selected_c2w,
                            depths_metric, gt_instances_2d,
                            max_depth=30.0):
    """Build GT-only first-camera normalized geometry for one scene."""
    if not np.isfinite(max_depth) or max_depth <= 0:
        raise ValueError('max_depth must be finite and positive')
    axis_align = as_homogeneous(axis_align_matrix, 'axis_align_matrix')
    selected_c2w = np.asarray(selected_c2w, dtype=np.float64)
    if (selected_c2w.ndim != 3 or selected_c2w.shape[1:] != (4, 4)
            or len(selected_c2w) == 0):
        raise ValueError('selected_c2w must have shape [V, 4, 4]')
    if len(selected_c2w) != len(depths_metric):
        raise ValueError('selected cameras and depth maps must share V')
    if len(gt_instances_2d) != len(selected_c2w):
        raise ValueError('2D instances and selected cameras must share V')

    points_aligned = transform_points(raw_points, axis_align)
    c2w_aligned = np.einsum('ij,vjk->vik', axis_align, selected_c2w)
    points_vggt, extrinsics_vggt, scale = (
        normalize_aligned_geometry_to_first_camera(
            points_aligned, c2w_aligned))
    c2w_vggt = np.stack([
        np.linalg.inv(as_homogeneous(extrinsic, 'normalized extrinsic'))
        for extrinsic in extrinsics_vggt
    ]).astype(np.float32)

    depths_metric = np.asarray(depths_metric, dtype=np.float32)
    valid_depth = (
        np.isfinite(depths_metric)
        & (depths_metric > 0)
        & (depths_metric < max_depth))
    depths_vggt = np.where(
        valid_depth, depths_metric / scale, 0.0).astype(np.float32)

    first_w2c_aligned = np.linalg.inv(c2w_aligned[0])
    for instances in gt_instances_2d:
        centers_aligned = np.asarray(
            instances.get('centers_3d', np.empty((0, 3))),
            dtype=np.float32).reshape(-1, 3)
        center_depth = np.asarray(
            instances.get('center_depth', np.empty((0,))),
            dtype=np.float32).reshape(-1)
        if len(centers_aligned) != len(center_depth):
            raise ValueError('2D centers and center depths must share N')
        center_valid = np.isfinite(centers_aligned).all(axis=1)
        center_depth_valid = (
            np.isfinite(center_depth)
            & (center_depth > 0)
            & (center_depth < max_depth))
        centers_vggt = np.full_like(centers_aligned, np.nan)
        if center_valid.any():
            centers_vggt[center_valid] = (
                transform_points(
                    centers_aligned[center_valid], first_w2c_aligned)
                / scale).astype(np.float32)
        center_depth_vggt = np.full_like(center_depth, np.nan)
        center_depth_vggt[center_depth_valid] = (
            center_depth[center_depth_valid] / scale)
        instances['centers_3d_aligned'] = centers_aligned
        instances['centers_3d_vggt'] = centers_vggt
        instances['center_depth_metric'] = center_depth
        instances['center_depth_vggt'] = center_depth_vggt
        instances['center_3d_valid_mask'] = center_valid
        instances['center_depth_valid_mask'] = center_depth_valid

    return {
        'gt_scene_points_aligned': points_aligned.astype(np.float32),
        'gt_scene_points_vggt': points_vggt,
        'gt_depths_vggt': depths_vggt,
        'gt_depth_valid_masks': valid_depth,
        'gt_extrinsics_vggt': extrinsics_vggt,
        'gt_c2w_vggt': c2w_vggt,
        'vggt_gt_scale': scale,
    }


@TRANSFORMS.register_module()
class BuildVGGTGroundTruth(BaseTransform):

    def __init__(self, points_root, num_point_features=6,
                 max_depth=30.0):
        if num_point_features < 3:
            raise ValueError('num_point_features must be at least 3')
        self.points_root = Path(points_root)
        self.num_point_features = int(num_point_features)
        self.max_depth = float(max_depth)

    def transform(self, results):
        scene_id = results.get('scene_id')
        if not scene_id:
            image_paths = results.get('img_path', [])
            if not image_paths:
                raise ValueError('Cannot determine scene id for VGGT GT')
            scene_id = Path(image_paths[0]).parent.name
        point_path = self.points_root / f'{scene_id}.bin'
        raw = np.fromfile(point_path, dtype=np.float32)
        if raw.size == 0 or raw.size % self.num_point_features:
            raise ValueError(f'Unexpected point cloud shape in {point_path}')
        raw_points = raw.reshape(-1, self.num_point_features)[:, :3]

        targets = build_vggt_ground_truth(
            raw_points=raw_points,
            axis_align_matrix=results['axis_align_matrix'],
            selected_c2w=results['gt_c2w_metric'],
            depths_metric=results['gt_depths_metric'],
            gt_instances_2d=results.get('gt_instances_2d', [
                {} for _ in results['gt_c2w_metric']]),
            max_depth=self.max_depth)
        results.update(targets)
        results['scene_id'] = str(scene_id)
        return results
