"""Generate ScanNet 2D boxes from visible projections of 3D instance points."""

from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import logging
import pickle
from pathlib import Path
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


LOGGER = logging.getLogger('scannet_3d_to_coco_bbox')
VISIBLE = np.int8(1)
OCCLUDED = np.int8(0)
UNKNOWN = np.int8(-1)


@dataclass(frozen=True)
class GeneratorConfig:
    depth_scale: float = 1000.0
    abs_depth_tolerance: float = 0.05
    rel_depth_tolerance: float = 0.01
    depth_window_radius: int = 1
    min_visible_points: int = 3
    min_visible_ratio: float = 0.2
    bbox_padding: float = 2.0
    center_min_samples: int = 3
    center_window_fraction: float = 0.1
    center_window_min_size: int = 2
    center_window_max_size: int = 6


@dataclass(frozen=True)
class PointAssignments:
    membership: np.ndarray
    owner: np.ndarray


@dataclass(frozen=True)
class SceneObject:
    instance_id: int
    label: int
    category_name: str
    box: np.ndarray
    point_indices: np.ndarray
    exclusive: np.ndarray


@dataclass(frozen=True)
class InstanceResult:
    instance_id: int
    category_id: int
    category_name: str
    bbox: Optional[list]
    amodal_bbox: Optional[list]
    visible_point_count: int
    visible_point_ratio: float
    truncated: bool
    rejection_reason: Optional[str] = None
    center_depth: Optional[float] = None
    center_3d: Optional[list] = None
    center_3d_sample_count: int = 0
    center_3d_source: Optional[str] = None

    @property
    def source(self) -> str:
        return 'visible_points' if self.bbox is not None else 'rejected'


@dataclass(frozen=True)
class ViewResult:
    view_index: int
    image_id: int
    file_name: str
    width: int
    height: int
    results: Dict[int, InstanceResult]


@dataclass(frozen=True)
class ViewGeometry:
    pixels: np.ndarray
    depth_pixels: np.ndarray
    depth: np.ndarray
    in_image: np.ndarray
    visibility: np.ndarray
    depth_map: np.ndarray
    depth_intrinsic: np.ndarray


@dataclass(frozen=True)
class VisualizationItem:
    bbox_xywh: Sequence[float]
    category_name: str
    instance_id: int
    visible_point_ratio: float
    category_id: int


_PALETTE = (
    (230, 57, 70), (29, 53, 87), (69, 123, 157), (42, 157, 143),
    (244, 162, 97), (233, 196, 106), (138, 201, 38), (255, 89, 94),
    (106, 76, 147), (25, 130, 196), (255, 146, 76), (87, 117, 144),
    (67, 170, 139), (249, 199, 79), (249, 65, 68), (144, 190, 109),
    (249, 132, 74), (39, 125, 161),
)


def camera_matrix_for_view(cam2img: np.ndarray, view_index: int) -> np.ndarray:
    matrices = np.asarray(cam2img, dtype=np.float64)
    if matrices.ndim == 3:
        if not 0 <= view_index < len(matrices):
            raise IndexError(f'view_index {view_index} is outside cam2img')
        matrices = matrices[view_index]
    if matrices.shape not in ((3, 3), (4, 4)):
        raise ValueError(f'cam2img must be 3x3, 4x4, or Vx..., got {matrices.shape}')
    return matrices[:3, :3]


def resize_intrinsic(intrinsic: np.ndarray, source_size: Tuple[int, int],
                     target_size: Tuple[int, int]) -> np.ndarray:
    """Scale a pinhole intrinsic from (width, height) to another image size."""
    source_width, source_height = map(float, source_size)
    target_width, target_height = map(float, target_size)
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError('image sizes must be positive')
    scaled = np.asarray(intrinsic, dtype=np.float64).copy()
    if scaled.shape != (3, 3):
        raise ValueError(f'intrinsic must be 3x3, got {scaled.shape}')
    scaled[0] *= target_width / source_width
    scaled[1] *= target_height / source_height
    return scaled


def load_depth_map(root: Path, image_path: Path, depth_scale: float = 1000.0) -> np.ndarray:
    """Load the matching ScanNet depth PNG and convert its units to metres."""
    if depth_scale <= 0 or not np.isfinite(depth_scale):
        raise ValueError('depth_scale must be finite and positive')
    depth_path = root / image_path.parent / 'depth' / f'{image_path.stem}.png'
    if not depth_path.is_file():
        raise FileNotFoundError(f'matching depth map not found: {depth_path}')
    with Image.open(depth_path) as depth_image:
        depth = np.asarray(depth_image)
    if depth.ndim != 2:
        raise ValueError(f'depth map must be single-channel: {depth_path}')
    return depth.astype(np.float32) / float(depth_scale)


def world_to_camera_from_aligned_pose(
        axis_align: np.ndarray, raw_camera_to_world: np.ndarray) -> np.ndarray:
    axis_align = np.asarray(axis_align, dtype=np.float64)
    raw_camera_to_world = np.asarray(raw_camera_to_world, dtype=np.float64)
    if axis_align.shape != (4, 4) or raw_camera_to_world.shape != (4, 4):
        raise ValueError('axis_align and raw_camera_to_world must both be 4x4')
    return np.linalg.inv(axis_align @ raw_camera_to_world)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'points must have shape Nx3, got {points.shape}')
    if transform.shape not in ((3, 4), (4, 4)):
        raise ValueError(f'transform must be 3x4 or 4x4, got {transform.shape}')
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(
        points: np.ndarray, world_to_camera: np.ndarray,
        intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    camera_points = transform_points(points, world_to_camera)
    depth = camera_points[:, 2]
    pixels_h = camera_points @ np.asarray(intrinsic, dtype=np.float64)[:3, :3].T
    with np.errstate(divide='ignore', invalid='ignore'):
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return pixels, depth


def points_in_image(
        pixels: np.ndarray, depth: np.ndarray, width: int, height: int,
        near: float = 1e-4) -> np.ndarray:
    pixels = np.asarray(pixels)
    depth = np.asarray(depth)
    return (
        (depth > near) & np.isfinite(depth) & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))


def _aabb_corners(box: np.ndarray) -> np.ndarray:
    center = np.asarray(box, dtype=np.float64)[:3]
    half_size = np.asarray(box, dtype=np.float64)[3:6] / 2.0
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    return center + signs * half_size


_AABB_EDGES = tuple(
    (first, second) for first in range(8) for second in range(first + 1, 8)
    if bin(first ^ second).count('1') == 1)


def project_aabb(
        box: np.ndarray, world_to_camera: np.ndarray, intrinsic: np.ndarray,
        width: int, height: int, near: float = 0.05) -> Optional[np.ndarray]:
    camera_corners = transform_points(_aabb_corners(box), world_to_camera)
    clipped = []
    for first_index, second_index in _AABB_EDGES:
        first = camera_corners[first_index]
        second = camera_corners[second_index]
        first_front = first[2] >= near
        second_front = second[2] >= near
        if first_front:
            clipped.append(first)
        if second_front:
            clipped.append(second)
        if first_front != second_front:
            ratio = (near - first[2]) / (second[2] - first[2])
            clipped.append(first + ratio * (second - first))
    if not clipped:
        return None
    camera_points = np.asarray(clipped)
    pixels_h = camera_points @ np.asarray(intrinsic)[:3, :3].T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    if not len(pixels):
        return None
    lower = np.maximum(pixels.min(axis=0), [0.0, 0.0])
    upper = np.minimum(pixels.max(axis=0), [float(width), float(height)])
    if np.any(upper <= lower):
        return None
    return np.concatenate([lower, upper]).astype(np.float32)


def classify_depth_visibility(
        pixels: np.ndarray, depth: np.ndarray, depth_map: np.ndarray,
        abs_tolerance: float = 0.05, rel_tolerance: float = 0.01,
        window_radius: int = 1) -> np.ndarray:
    """Classify projected points against the measured depth surface.

    A point is visible when its camera-space depth is no farther than the
    nearest valid depth sample in a small pixel neighbourhood. The operation
    is per point and never splits an instance into connected components.
    """
    if window_radius < 0:
        raise ValueError('window_radius must be non-negative')
    pixels = np.asarray(pixels, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    depth_map = np.asarray(depth_map, dtype=np.float64)
    if depth_map.ndim != 2:
        raise ValueError('depth_map must have shape [H, W]')
    height, width = depth_map.shape
    result = np.full(len(pixels), UNKNOWN, dtype=np.int8)
    valid = ((depth > 1e-4) & np.isfinite(depth)
             & np.isfinite(pixels).all(axis=1))
    for point_index in np.flatnonzero(valid):
        x, y = np.rint(pixels[point_index]).astype(np.int64)
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        x0, x1 = max(0, x - window_radius), min(width, x + window_radius + 1)
        y0, y1 = max(0, y - window_radius), min(height, y + window_radius + 1)
        samples = depth_map[y0:y1, x0:x1]
        samples = samples[np.isfinite(samples) & (samples > 1e-4)]
        if samples.size == 0:
            continue
        tolerance = abs_tolerance + rel_tolerance * depth[point_index]
        result[point_index] = int(depth[point_index] <= samples.min() + tolerance)
    return result


def assign_scene_points(points: np.ndarray, boxes: np.ndarray) -> PointAssignments:
    points = np.asarray(points, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return PointAssignments(
            np.zeros((len(points), 0), dtype=bool),
            np.full(len(points), -1, dtype=np.int32))
    lower = boxes[:, :3] - boxes[:, 3:6] / 2.0
    upper = boxes[:, :3] + boxes[:, 3:6] / 2.0
    membership = np.all(
        (points[:, None, :] >= lower[None])
        & (points[:, None, :] <= upper[None]), axis=2)
    counts = membership.sum(axis=1)
    owner = np.full(len(points), -1, dtype=np.int32)
    unique = counts == 1
    owner[unique] = membership[unique].argmax(axis=1).astype(np.int32)
    return PointAssignments(membership, owner)


def visible_points_bbox(
        pixels: np.ndarray, amodal_xyxy: np.ndarray, width: int, height: int,
        padding: float = 2.0) -> Optional[list]:
    """Return one xywh box enclosing all visible projections."""
    pixels = np.asarray(pixels, dtype=np.float64)
    if not len(pixels):
        return None
    lower = pixels.min(axis=0) - padding
    upper = pixels.max(axis=0) + padding
    amodal = np.asarray(amodal_xyxy, dtype=np.float64)
    lower = np.maximum(lower, amodal[:2])
    upper = np.minimum(upper, amodal[2:])
    lower = np.maximum(lower, [0.0, 0.0])
    upper = np.minimum(upper, [float(width), float(height)])
    if np.any(upper <= lower):
        return None
    return [float(lower[0]), float(lower[1]),
            float(upper[0] - lower[0]), float(upper[1] - lower[1])]


def sample_view_indices(
        total: int, requested: int, strategy: str,
        rng: np.random.Generator) -> np.ndarray:
    if total < 0 or requested == 0:
        raise ValueError('total must be non-negative and requested cannot be zero')
    count = total if requested < 0 else min(total, requested)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if strategy == 'uniform':
        return np.rint(np.linspace(0, total - 1, count)).astype(np.int64)
    if strategy == 'random':
        return np.sort(rng.choice(total, count, replace=False)).astype(np.int64)
    raise ValueError(f'unknown sampling strategy: {strategy}')


def _resolve_point_path(root: Path, relative: Path) -> Path:
    candidates = (relative, root / relative, root / 'points' / relative.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        'point cloud not found; checked: ' + ', '.join(map(str, candidates)))


def load_axis_aligned_scene_points(root: Path, info: dict) -> np.ndarray:
    metadata = info.get('lidar_points', {})
    dimensions = int(metadata.get('num_pts_feats', 6))
    aligned_relative = info.get('aligned_pts_path')
    if aligned_relative is not None:
        relative = Path(aligned_relative)
    elif metadata.get('lidar_path'):
        relative = Path(metadata['lidar_path'])
    else:
        scene = Path(info['img_paths'][0]).parent.name
        relative = Path('points') / f'{scene}.bin'
    path = _resolve_point_path(Path(root), relative)
    values = np.fromfile(path, dtype=np.float32)
    if dimensions < 3 or values.size % dimensions:
        raise ValueError(f'invalid {dimensions}-D point file: {path}')
    points = values.reshape(-1, dimensions)[:, :3]
    if aligned_relative is not None:
        return points
    axis = np.asarray(info['axis_align_matrix'], dtype=np.float32)
    if axis.shape != (4, 4):
        raise ValueError(f'axis_align_matrix must be 4x4, got {axis.shape}')
    return points @ axis[:3, :3].T + axis[:3, 3]


def annotation_from_bbox(
        bbox: Sequence[float], annotation_id: int, image_id: int,
        category_id: int, instance_id_3d: int, extra: dict) -> dict:
    bbox = [float(value) for value in bbox]
    if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError('bbox must be valid xywh')
    annotation = {
        'id': int(annotation_id), 'image_id': int(image_id),
        'category_id': int(category_id), 'bbox': bbox,
        'area': float(bbox[2] * bbox[3]), 'iscrowd': 0,
        'instance_id_3d': int(instance_id_3d),
    }
    annotation.update(extra)
    return annotation


def _xyxy_to_xywh(box: Optional[np.ndarray]) -> Optional[list]:
    if box is None:
        return None
    return [float(box[0]), float(box[1]),
            float(box[2] - box[0]), float(box[3] - box[1])]


def _is_truncated(box: np.ndarray, width: int, height: int) -> bool:
    return bool(box[0] <= 0.5 or box[1] <= 0.5
                or box[2] >= width - 0.5 or box[3] >= height - 0.5)


def _prepare_objects(
        info: dict, points: np.ndarray,
        category_by_label: Dict[int, str]) -> Tuple[list, PointAssignments]:
    instances = info.get('instances', [])
    boxes = np.asarray(
        [instance['bbox_3d'][:6] for instance in instances], dtype=np.float32)
    if not len(instances):
        boxes = np.empty((0, 6), dtype=np.float32)
    assignments = assign_scene_points(points, boxes)
    objects = []
    seen_instance_ids = set()
    for list_index, instance in enumerate(instances):
        if 'instance_id' in instance:
            instance_id = int(instance['instance_id'])
        else:
            instance_id = list_index
        if instance_id in seen_instance_ids:
            raise ValueError(f'duplicate instance_id in scene: {instance_id}')
        seen_instance_ids.add(instance_id)
        label = int(instance['bbox_label_3d'])
        if label not in category_by_label:
            raise ValueError(f'unknown bbox_label_3d {label}')
        indices = np.flatnonzero(assignments.membership[:, list_index])
        exclusive = assignments.owner[indices] == list_index
        box = boxes[list_index]
        lower_z = box[2] - box[5] / 2.0
        support_band = min(0.05, 0.1 * float(box[5]))
        exclusive &= points[indices, 2] > lower_z + support_band
        objects.append(SceneObject(
            instance_id, label, category_by_label[label], box,
            indices, exclusive))
    return objects, assignments


def _build_view_geometry(
        points: np.ndarray, world_to_camera: np.ndarray, intrinsic: np.ndarray,
        width: int, height: int, depth_map: np.ndarray,
        config: GeneratorConfig) -> ViewGeometry:
    depth_height, depth_width = depth_map.shape
    depth_intrinsic = resize_intrinsic(
        intrinsic, (width, height), (depth_width, depth_height))
    depth_pixels, point_depth = project_points(
        points, world_to_camera, depth_intrinsic)
    in_depth = points_in_image(
        depth_pixels, point_depth, depth_width, depth_height)
    visibility = classify_depth_visibility(
        depth_pixels, point_depth, depth_map,
        config.abs_depth_tolerance, config.rel_depth_tolerance,
        config.depth_window_radius)
    pixels, _ = project_points(points, world_to_camera, intrinsic)
    in_image = points_in_image(pixels, point_depth, width, height)
    visibility[in_depth == 0] = UNKNOWN
    scale = np.asarray([width / depth_width, height / depth_height])
    pixels = depth_pixels * scale
    return ViewGeometry(
        pixels=pixels,
        depth_pixels=depth_pixels,
        depth=point_depth,
        in_image=in_image,
        visibility=visibility,
        depth_map=depth_map,
        depth_intrinsic=depth_intrinsic)


def _robust_depth_from_patch(depth_map: np.ndarray, center: np.ndarray,
                             window_size: int) -> Optional[float]:
    if window_size <= 0:
        raise ValueError('window_size must be positive')
    height, width = depth_map.shape
    x, y = np.rint(center).astype(np.int64)
    if x < 0 or x >= width or y < 0 or y >= height:
        return None
    left = window_size // 2
    right = window_size - left
    x0, x1 = max(0, x - left), min(width, x + right)
    y0, y1 = max(0, y - left), min(height, y + right)
    values = depth_map[y0:y1, x0:x1].astype(np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 1e-4)]
    if not len(values):
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    tolerance = max(0.05, 3.0 * 1.4826 * mad)
    inliers = values[np.abs(values - median) <= tolerance]
    return float(np.median(inliers)) if len(inliers) else median


def _backproject_depth_pixel(pixel: np.ndarray, depth: float,
                             intrinsic: np.ndarray,
                             world_to_camera: np.ndarray) -> np.ndarray:
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    camera_point = np.array([
        (pixel[0] - cx) * depth / fx,
        (pixel[1] - cy) * depth / fy,
        depth,
    ], dtype=np.float64)
    transform = np.asarray(world_to_camera, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (camera_point - translation) @ rotation


def _estimate_center_geometry(
        obj: SceneObject, points: np.ndarray, geometry: ViewGeometry,
        bbox: Sequence[float], world_to_camera: np.ndarray, width: int,
        height: int, config: GeneratorConfig) -> Tuple[Optional[float],
                                                        Optional[list], int,
                                                        Optional[str]]:
    x, y, box_width, box_height = map(float, bbox)
    center_rgb = np.array([x + box_width / 2.0, y + box_height / 2.0])
    depth_height, depth_width = geometry.depth_map.shape
    center_depth_pixel = center_rgb * np.array([
        depth_width / float(width), depth_height / float(height)])
    depth_box_size = min(
        box_width * depth_width / float(width),
        box_height * depth_height / float(height))
    window_size = int(np.clip(
        np.rint(2.0 * config.center_window_fraction * depth_box_size + 1.0),
        config.center_window_min_size, config.center_window_max_size))
    center_depth = _robust_depth_from_patch(
        geometry.depth_map, center_depth_pixel, window_size)

    indices = obj.point_indices
    visible = ((geometry.visibility[indices] == VISIBLE)
               & geometry.in_image[indices])
    projected = geometry.pixels[indices]
    pixel_radius = max(
        2.0,
        config.center_window_fraction * min(box_width, box_height))
    near_center = np.linalg.norm(projected - center_rgb[None], axis=1)
    center_point_mask = visible & (near_center <= pixel_radius)
    center_points = points[indices][center_point_mask]
    if len(center_points) >= config.center_min_samples:
        mean_point = np.mean(center_points.astype(np.float64), axis=0)
        return (center_depth, mean_point.astype(np.float32).tolist(),
                int(len(center_points)), 'instance_points')

    if center_depth is not None:
        point = _backproject_depth_pixel(
            center_depth_pixel, center_depth, geometry.depth_intrinsic,
            world_to_camera)
        return center_depth, point.astype(np.float32).tolist(), 0, 'depth_backprojection'

    visible_depths = geometry.depth[indices][visible]
    if len(visible_depths):
        fallback_depth = float(np.median(visible_depths))
        point = _backproject_depth_pixel(
            center_depth_pixel, fallback_depth, geometry.depth_intrinsic,
            world_to_camera)
        return (fallback_depth, point.astype(np.float32).tolist(), 0,
                'visible_point_backprojection')
    return None, None, 0, None


def _infer_object(
        obj: SceneObject, points: np.ndarray, geometry: ViewGeometry,
        world_to_camera: np.ndarray, intrinsic: np.ndarray,
        width: int, height: int, config: GeneratorConfig) -> InstanceResult:
    amodal = project_aabb(obj.box, world_to_camera, intrinsic, width, height)
    if amodal is None:
        return InstanceResult(
            obj.instance_id, obj.label + 1, obj.category_name, None, None,
            0, 0.0, False, 'outside_camera_frustum')
    if not len(obj.point_indices):
        return InstanceResult(
            obj.instance_id, obj.label + 1, obj.category_name, None,
            _xyxy_to_xywh(amodal), 0, 0.0,
            _is_truncated(amodal, width, height), 'aabb_contains_no_points')

    indices = obj.point_indices
    visible = ((geometry.visibility[indices] == VISIBLE)
               & geometry.in_image[indices])
    visible_count = int(visible.sum())
    visible_ratio = visible_count / len(indices)
    if visible_ratio <= config.min_visible_ratio:
        return InstanceResult(
            obj.instance_id, obj.label + 1, obj.category_name, None,
            _xyxy_to_xywh(amodal), visible_count, visible_ratio,
            _is_truncated(amodal, width, height), 'mostly_occluded')
    if visible_count < config.min_visible_points:
        return InstanceResult(
            obj.instance_id, obj.label + 1, obj.category_name, None,
            _xyxy_to_xywh(amodal), visible_count, visible_ratio,
            _is_truncated(amodal, width, height), 'too_few_visible_points')

    exclusive_visible = visible & obj.exclusive
    bbox_points = geometry.pixels[indices][exclusive_visible]
    if len(bbox_points) < config.min_visible_points:
        bbox_points = geometry.pixels[indices][visible]
    bbox = visible_points_bbox(
        bbox_points, amodal, width, height,
        padding=config.bbox_padding)
    if bbox is None:
        return InstanceResult(
            obj.instance_id, obj.label + 1, obj.category_name, None,
            _xyxy_to_xywh(amodal), visible_count, visible_ratio,
            _is_truncated(amodal, width, height), 'invalid_visible_bbox')
    center_depth, center_3d, center_sample_count, center_source = (
        _estimate_center_geometry(
            obj, points, geometry, bbox, world_to_camera, width, height,
            config))
    return InstanceResult(
        obj.instance_id, obj.label + 1, obj.category_name, bbox,
        _xyxy_to_xywh(amodal), visible_count, visible_ratio,
        _is_truncated(amodal, width, height),
        None if bbox is not None else 'invalid_visible_bbox',
        center_depth, center_3d, center_sample_count, center_source)


def _process_view(
        root: Path, info: dict, points: np.ndarray,
        objects: Iterable[SceneObject], view_index: int, image_id: int,
        config: GeneratorConfig) -> ViewResult:
    relative = Path(info['img_paths'][view_index])
    image_path = relative if relative.is_absolute() else root / relative
    with Image.open(image_path) as image:
        width, height = image.size
    depth_map = load_depth_map(root, relative, config.depth_scale)
    world_to_camera = world_to_camera_from_aligned_pose(
        np.asarray(info['axis_align_matrix'], dtype=np.float64),
        np.asarray(info['lidar2cam'][view_index]))
    intrinsic = camera_matrix_for_view(info['cam2img'], view_index)
    geometry = _build_view_geometry(
        points, world_to_camera, intrinsic, width, height, depth_map, config)
    results = {
        obj.instance_id: _infer_object(
            obj, points, geometry, world_to_camera, intrinsic,
            width, height, config)
        for obj in objects
    }
    return ViewResult(
        view_index, image_id, str(relative), width, height, results)


def render_annotation_image(
        image: Image.Image, items: Iterable[VisualizationItem]) -> Image.Image:
    rendered = image.convert('RGB')
    draw = ImageDraw.Draw(rendered)
    width, height = rendered.size
    for item in items:
        color = _PALETTE[(int(item.category_id) - 1) % len(_PALETTE)]
        x, y, box_width, box_height = map(float, item.bbox_xywh)
        x2 = min(float(width - 1), x + box_width)
        y2 = min(float(height - 1), y + box_height)
        draw.rectangle((x, y, x2, y2), outline=color, width=3)
        label = (f'{item.category_name} #{item.instance_id} '
                 f'visible={item.visible_point_ratio:.2f}')
        text_box = draw.textbbox((0, 0), label)
        label_width = min(text_box[2] - text_box[0] + 6, width)
        label_height = text_box[3] - text_box[1] + 4
        label_x = max(0, min(int(x), width - label_width))
        label_y = max(0, int(y) - label_height)
        draw.rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            fill=color)
        draw.text((label_x + 3, label_y + 1), label, fill=(255, 255, 255))
    return rendered


def _save_view_visualization(
        root: Path, visualization_root: Path, scene_id: str,
        view: ViewResult) -> Path:
    relative = Path(view.file_name)
    image_path = relative if relative.is_absolute() else root / relative
    with Image.open(image_path) as source_image:
        image = source_image.convert('RGB')
    items = [
        VisualizationItem(
            result.bbox, result.category_name, result.instance_id,
            result.visible_point_ratio, result.category_id)
        for result in view.results.values() if result.bbox is not None
    ]
    output = visualization_root / scene_id / f'{relative.stem}_annotations.jpg'
    output.parent.mkdir(parents=True, exist_ok=True)
    render_annotation_image(image, items).save(output, quality=92)
    return output


def _scene_id(info: dict) -> str:
    if info.get('scene_id'):
        return str(info['scene_id'])
    if info.get('lidar_points', {}).get('lidar_path'):
        return Path(info['lidar_points']['lidar_path']).stem
    return Path(info['img_paths'][0]).parent.name


def convert(args: argparse.Namespace) -> None:
    started_at = time.perf_counter()
    root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    annotation_path = Path(args.ann_file)
    if not annotation_path.is_absolute():
        annotation_path = root / annotation_path
    with annotation_path.open('rb') as handle:
        payload = pickle.load(handle)
    data_list = payload['data_list']
    if args.max_scenes >= 0:
        data_list = data_list[:args.max_scenes]
    categories = sorted(
        payload['metainfo']['categories'].items(), key=lambda item: item[1])
    category_by_label = {int(label): name for name, label in categories}
    coco = {
        'info': {'description':
                 'ScanNet boxes from visible 3D instance point projections'},
        'licenses': [], 'images': [], 'annotations': [],
        'categories': [
            {'id': label + 1, 'name': name, 'supercategory': 'object'}
            for name, label in categories],
    }
    config = GeneratorConfig(
        depth_scale=args.depth_scale,
        abs_depth_tolerance=args.abs_depth_tolerance,
        rel_depth_tolerance=args.rel_depth_tolerance,
        depth_window_radius=args.depth_window_radius,
        min_visible_points=args.min_visible_points,
        min_visible_ratio=args.min_visible_ratio,
        bbox_padding=args.bbox_padding,
        center_min_samples=args.center_min_samples,
        center_window_fraction=args.center_window_fraction,
        center_window_min_size=args.center_window_min_size,
        center_window_max_size=args.center_window_max_size)
    visualization_root = None
    if args.visualize:
        visualization_root = Path(args.visualization_dir).resolve() \
            if args.visualization_dir else output.with_name(
                output.stem + '_visualizations')
        LOGGER.info('Visualization enabled: output=%s', visualization_root)

    rng = np.random.default_rng(args.seed)
    stats = Counter()
    rejections = []
    image_id = annotation_id = 1
    visualized_images = 0
    LOGGER.info(
        'Starting annotation generation: scenes=%d views_per_scene=%d sampling=%s',
        len(data_list), args.num_views, args.sampling)

    for scene_offset, info in enumerate(data_list):
        scene_started_at = time.perf_counter()
        points = load_axis_aligned_scene_points(root, info)
        objects, _ = _prepare_objects(info, points, category_by_label)
        view_indices = sample_view_indices(
            len(info['img_paths']), args.num_views, args.sampling, rng)
        scene_id = _scene_id(info)
        LOGGER.info(
            'Scene %d/%d %s: points=%d objects=%d selected_views=%d/%d',
            scene_offset + 1, len(data_list), scene_id, len(points),
            len(objects), len(view_indices), len(info['img_paths']))

        for selected_offset, view_index in enumerate(view_indices):
            view_started_at = time.perf_counter()
            view = _process_view(
                root, info, points, objects, int(view_index), image_id, config)
            coco['images'].append({
                'id': image_id, 'file_name': view.file_name,
                'width': view.width, 'height': view.height,
                'scene_id': scene_id, 'view_index': view.view_index})
            for result in view.results.values():
                if result.bbox is None:
                    stats['rejected'] += 1
                    rejections.append({
                        'scene_id': scene_id, 'view_index': view.view_index,
                        'instance_id_3d': result.instance_id,
                        'category_name': result.category_name,
                        'reason': result.rejection_reason})
                    continue
                coco['annotations'].append(annotation_from_bbox(
                    result.bbox, annotation_id, image_id,
                    result.category_id, result.instance_id, {
                        'scene_id': scene_id,
                        'bbox_amodal': result.amodal_bbox,
                        'visible_point_count': result.visible_point_count,
                        'visible_point_ratio': result.visible_point_ratio,
                        'truncated': result.truncated,
                        'annotation_source': result.source,
                        'center_depth': result.center_depth,
                        'center_3d': result.center_3d,
                        'center_3d_sample_count': result.center_3d_sample_count,
                        'center_3d_source': result.center_3d_source,
                    }))
                annotation_id += 1
                stats['visible_points'] += 1
            if (visualization_root is not None
                    and (args.visualization_max_images < 0
                         or visualized_images < args.visualization_max_images)):
                path = _save_view_visualization(
                    root, visualization_root, scene_id, view)
                visualized_images += 1
                LOGGER.debug('Saved visualization %d: %s', visualized_images, path)
            image_id += 1
            stats['images'] += 1
            if (selected_offset == 0
                    or (selected_offset + 1) % args.view_log_interval == 0
                    or selected_offset + 1 == len(view_indices)):
                view_counts = Counter(r.source for r in view.results.values())
                LOGGER.info(
                    'Scene %s view %d/%d frame=%d: accepted=%d rejected=%d '
                    'time=%.2fs', scene_id, selected_offset + 1,
                    len(view_indices), int(view_index),
                    view_counts['visible_points'], view_counts['rejected'],
                    time.perf_counter() - view_started_at)
        stats['scenes'] += 1
        LOGGER.info('Scene %s complete: images=%d elapsed=%.1fs',
                    scene_id, len(view_indices),
                    time.perf_counter() - scene_started_at)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, separators=(',', ':')))
    rejection_output = Path(args.rejections_output) if args.rejections_output \
        else output.with_name(output.stem + '_rejections.json')
    rejection_output.write_text(json.dumps(rejections, indent=2))
    LOGGER.info('Wrote %d annotations to %s', len(coco['annotations']), output)
    LOGGER.info('Wrote %d rejected candidates to %s', len(rejections), rejection_output)
    if visualization_root is not None:
        LOGGER.info('Wrote %d visualization images to %s',
                    visualized_images, visualization_root)
    LOGGER.info('Stats: %s', dict(stats))
    LOGGER.info('Finished in %.1f seconds', time.perf_counter() - started_at)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-views', type=int, default=50)
    parser.add_argument('--sampling', choices=('uniform', 'random'), default='uniform')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-scenes', type=int, default=-1)
    parser.add_argument('--abs-depth-tolerance', type=float, default=0.05)
    parser.add_argument('--rel-depth-tolerance', type=float, default=0.01)
    parser.add_argument('--depth-scale', type=float, default=1000.0)
    parser.add_argument('--depth-window-radius', type=int, default=1)
    parser.add_argument('--min-visible-points', type=int, default=3)
    parser.add_argument(
        '--min-visible-ratio', type=float, default=0.2,
        help='Reject an instance when less than this fraction of its 3D points is visible.')
    parser.add_argument('--center-min-samples', type=int, default=3)
    parser.add_argument('--center-window-fraction', type=float, default=0.1)
    parser.add_argument('--center-window-min-size', type=int, default=2)
    parser.add_argument('--center-window-max-size', type=int, default=6)
    parser.add_argument('--bbox-padding', type=float, default=2.0)
    parser.add_argument('--rejections-output')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--visualization-dir')
    parser.add_argument('--visualization-max-images', type=int, default=-1)
    parser.add_argument('--view-log-interval', type=int, default=10)
    parser.add_argument(
        '--log-level', choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
        default='INFO')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for field in ('min_visible_points', 'view_log_interval'):
        if getattr(args, field) <= 0:
            raise ValueError(f'--{field.replace("_", "-")} must be positive')
    if args.depth_scale <= 0:
        raise ValueError('--depth-scale must be positive')
    if args.depth_window_radius < 0:
        raise ValueError('--depth-window-radius must be non-negative')
    if not 0.0 <= args.min_visible_ratio <= 1.0:
        raise ValueError('--min-visible-ratio must be in [0, 1]')
    if args.center_min_samples <= 0:
        raise ValueError('--center-min-samples must be positive')
    if args.center_window_fraction <= 0:
        raise ValueError('--center-window-fraction must be positive')
    if args.center_window_min_size <= 0:
        raise ValueError('--center-window-min-size must be positive')
    if args.center_window_max_size < args.center_window_min_size:
        raise ValueError('--center-window-max-size must be >= min size')
    if args.bbox_padding < 0:
        raise ValueError('--bbox-padding must be non-negative')
    if args.visualization_max_images == 0 or args.visualization_max_images < -1:
        raise ValueError('--visualization-max-images must be -1 or positive')
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s | %(levelname)s | %(message)s')
    convert(args)


if __name__ == '__main__':
    main()
