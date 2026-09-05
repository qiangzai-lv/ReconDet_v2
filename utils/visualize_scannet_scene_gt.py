#!/usr/bin/env python3
"""Export one ScanNet scene's aligned GT geometry and annotated RGB views."""

import argparse
import json
import logging
import pickle
import random
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


LOGGER = logging.getLogger("visualize_scannet_scene_gt")
BOX_COLOR = np.array([40, 130, 255], dtype=np.uint8)
CENTER_COLOR = np.array([255, 0, 0], dtype=np.uint8)
FACE_COLOR = np.array([0, 220, 70], dtype=np.uint8)
BOX_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)


FACE_NORMALS = np.array([
    [-1, 0, 0], [1, 0, 0],
    [0, -1, 0], [0, 1, 0],
    [0, 0, -1], [0, 0, 1],
], dtype=np.float32)


def align_points(points: np.ndarray, axis_align_matrix: np.ndarray) -> np.ndarray:
    """Transform ScanNet points into the axis-aligned GT box coordinates."""
    matrix = np.asarray(axis_align_matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"axis_align_matrix must be 4x4, got {matrix.shape}")
    points = np.asarray(points, dtype=np.float32)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def box_geometry(center: np.ndarray,
                 size: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return eight AABB corners and six ordered face centers."""
    center = np.asarray(center, dtype=np.float32)
    half_size = np.asarray(size, dtype=np.float32) / 2.0
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float32)
    corners = center + signs * half_size
    face_centers = center + FACE_NORMALS * half_size
    return corners, face_centers


def sample_indices(total: int, count: int, mode: str, seed: int) -> List[int]:
    """Select at most count indices, uniformly or randomly."""
    if total <= 0 or count <= 0:
        return []
    count = min(total, count)
    if mode == "uniform":
        return np.linspace(0, total - 1, count, dtype=np.int64).tolist()
    if mode == "random":
        return sorted(random.Random(seed).sample(range(total), count))
    raise ValueError(f"Unknown sampling mode: {mode}")


def _line_points(start: np.ndarray, end: np.ndarray,
                 step: float) -> np.ndarray:
    length = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(length / step)) + 1)
    return np.linspace(start, end, count, dtype=np.float32)


def _sphere_points(center: np.ndarray, radius: float,
                   count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float32) + 0.5
    z = 1.0 - 2.0 * indices / count
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = np.pi * (3.0 - np.sqrt(5.0)) * indices
    unit = np.column_stack((radial * np.cos(angle),
                            radial * np.sin(angle), z))
    return center[None] + radius * unit.astype(np.float32)


def _colored(points: np.ndarray, color: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    colors = np.broadcast_to(color, (len(points), 3)).copy()
    return points.astype(np.float32, copy=False), colors


def build_scene_geometry(raw_points: np.ndarray, raw_colors: np.ndarray,
                         axis_align_matrix: np.ndarray, boxes: np.ndarray,
                         line_step: float, marker_radius: float,
                         marker_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Combine aligned RGB points, AABB edges, keypoints, and links."""
    if line_step <= 0 or marker_radius <= 0 or marker_samples < 4:
        raise ValueError("line_step and marker_radius must be positive; "
                         "marker_samples must be at least 4")
    point_parts = [align_points(raw_points, axis_align_matrix)]
    color_parts = [np.asarray(raw_colors, dtype=np.uint8)]
    for box in np.asarray(boxes, dtype=np.float32):
        center, size = box[:3], box[3:6]
        corners, faces = box_geometry(center, size)
        for first, second in BOX_EDGES:
            points, colors = _colored(
                _line_points(corners[first], corners[second], line_step),
                BOX_COLOR)
            point_parts.append(points)
            color_parts.append(colors)
        for face in faces:
            points, colors = _colored(
                _line_points(center, face, line_step), FACE_COLOR)
            point_parts.append(points)
            color_parts.append(colors)
        points, colors = _colored(
            _sphere_points(center, marker_radius, marker_samples), CENTER_COLOR)
        point_parts.append(points)
        color_parts.append(colors)
        for face in faces:
            points, colors = _colored(
                _sphere_points(face, marker_radius, marker_samples), FACE_COLOR)
            point_parts.append(points)
            color_parts.append(colors)
    return np.concatenate(point_parts), np.concatenate(color_parts)


def _annotation_keypoints(annotation: Dict[str, Any]
                          ) -> Tuple[List[Tuple[float, float]], List[int]]:
    flat = annotation.get("keypoints_2d")
    if flat is not None and len(flat) >= 8:
        points = [(float(flat[index]), float(flat[index + 1]))
                  for index in range(0, 8, 2)]
    else:
        keypoints = annotation.get("keypoints", [])
        if len(keypoints) < 12:
            return [], []
        points = [(float(keypoints[index]), float(keypoints[index + 1]))
                  for index in range(0, 12, 3)]
    visibility = annotation.get("keypoints_visibility")
    if visibility is None:
        keypoints = annotation.get("keypoints", [])
        visibility = ([int(keypoints[index + 2]) for index in range(0, 12, 3)]
                      if len(keypoints) >= 12 else [1] * 4)
    return points, [int(value) for value in visibility[:4]]


def draw_2d_annotations(image: Image.Image,
                        annotations: Iterable[Dict[str, Any]],
                        point_radius: int, line_width: int) -> Image.Image:
    """Draw red centers, green face centers, and thin links."""
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    for annotation in annotations:
        points, visibility = _annotation_keypoints(annotation)
        if len(points) != 4 or len(visibility) != 4:
            continue
        center = points[0]
        if visibility[0]:
            for face, visible in zip(points[1:], visibility[1:]):
                if visible:
                    draw.line((center, face), fill=tuple(FACE_COLOR),
                              width=max(1, line_width))
        radius = max(1, point_radius)
        for index, ((x, y), visible) in enumerate(zip(points, visibility)):
            if not visible:
                continue
            color = CENTER_COLOR if index == 0 else FACE_COLOR
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill=tuple(color))
    return output


def write_binary_ply(path: Path, points: np.ndarray,
                     colors: np.ndarray) -> None:
    """Write XYZRGB vertices as a compact, broadly supported binary PLY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype="<f4")
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and colors must both have shape (N, 3)")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode("ascii")
    vertex = struct.Struct("<fffBBB")
    with path.open("wb") as handle:
        handle.write(header)
        for point, color in zip(points, colors):
            handle.write(vertex.pack(float(point[0]), float(point[1]),
                                     float(point[2]), int(color[0]),
                                     int(color[1]), int(color[2])))


def _find_scene(data_list: Sequence[Dict[str, Any]],
                scene_id: str) -> Dict[str, Any]:
    for item in data_list:
        item_id = str(item.get("scene_id") or
                      Path(item["lidar_points"]["lidar_path"]).stem)
        if item_id == scene_id:
            return item
    raise KeyError(f"Scene {scene_id!r} was not found in the annotation file")


def _load_scene_info(path: Path, scene_id: str) -> Dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return _find_scene(payload["data_list"], scene_id)


def _load_gt_points(data_root: Path, info: Dict[str, Any]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    relative = Path(info["lidar_points"]["lidar_path"])
    candidates = (data_root / "points" / relative, data_root / relative)
    point_path = next((path for path in candidates if path.is_file()), candidates[0])
    dimensions = int(info["lidar_points"].get("num_pts_feats", 6))
    raw = np.fromfile(point_path, dtype=np.float32)
    if dimensions < 3 or raw.size % dimensions:
        raise ValueError(f"Unexpected point cloud shape in {point_path}")
    raw = raw.reshape(-1, dimensions)
    if dimensions >= 6:
        colors = raw[:, 3:6]
        if colors.max(initial=0.0) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        colors = np.full((len(raw), 3), 160, dtype=np.uint8)
    return raw[:, :3], colors


def _scene_images(coco: Dict[str, Any], scene_id: str
                  ) -> List[Dict[str, Any]]:
    images = []
    for image in coco.get("images", []):
        inferred = Path(str(image.get("file_name", ""))).parent.name
        if str(image.get("scene_id", inferred)) == scene_id:
            images.append(image)
    return sorted(images, key=lambda item: int(item.get("view_index", item["id"])))


def render_views(coco_path: Path, data_root: Path, scene_id: str,
                 output_dir: Path, num_views: int, sampling: str, seed: int,
                 point_radius: int, line_width: int) -> int:
    with coco_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    images = _scene_images(coco, scene_id)
    if not images:
        raise ValueError(f"No COCO images found for scene {scene_id!r}")
    by_image = defaultdict(list)
    for annotation in coco.get("annotations", []):
        by_image[annotation["image_id"]].append(annotation)
    selected = sample_indices(len(images), num_views, sampling, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for position in selected:
        record = images[position]
        source = Path(record["file_name"])
        if not source.is_absolute():
            source = data_root / source
        if not source.is_file():
            LOGGER.warning("Skipping missing image: %s", source)
            continue
        with Image.open(source) as image:
            rendered = draw_2d_annotations(
                image, by_image.get(record["id"], []), point_radius, line_width)
        view_index = int(record.get("view_index", record["id"]))
        rendered.save(output_dir / f"view_{view_index:05d}.jpg", quality=95)
        saved += 1
    return saved


def visualize_scene(args: argparse.Namespace) -> Tuple[Path, int]:
    data_root = Path(args.data_root).resolve()
    ann_path = Path(args.ann_file)
    if not ann_path.is_absolute():
        ann_path = data_root / ann_path
    info = _load_scene_info(ann_path, args.scene_id)
    raw_points, raw_colors = _load_gt_points(data_root, info)
    boxes = np.asarray([instance["bbox_3d"][:6]
                        for instance in info.get("instances", [])],
                       dtype=np.float32).reshape(-1, 6)
    points, colors = build_scene_geometry(
        raw_points, raw_colors, np.asarray(info["axis_align_matrix"]), boxes,
        args.line_step, args.marker_radius, args.marker_samples)

    scene_dir = Path(args.output_dir) / args.scene_id
    ply_path = scene_dir / "scene_gt_annotations.ply"
    write_binary_ply(ply_path, points, colors)
    view_count = render_views(
        Path(args.coco_json), data_root, args.scene_id, scene_dir / "views",
        args.num_views, args.sampling, args.seed, args.point_radius,
        args.line_width)
    LOGGER.info("Wrote %s with %d vertices (%d GT points, %d boxes)",
                ply_path, len(points), len(raw_points), len(boxes))
    LOGGER.info("Wrote %d annotated views to %s", view_count,
                scene_dir / "views")
    return ply_path, view_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--data-root",
                        default="/root/shared-nvme/data/ScanNet_processed")
    parser.add_argument("--ann-file", default="scannet_infos_val_pts.pkl")
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--output-dir", default="vis/scannet_scene_gt")
    parser.add_argument("--num-views", type=int, default=100)
    parser.add_argument("--sampling", choices=("uniform", "random"),
                        default="uniform")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--line-step", type=float, default=0.015,
                        help="3D spacing for sampled box/link lines in meters")
    parser.add_argument("--marker-radius", type=float, default=0.04,
                        help="3D keypoint sphere radius in meters")
    parser.add_argument("--marker-samples", type=int, default=96)
    parser.add_argument("--point-radius", type=int, default=5,
                        help="2D keypoint radius in pixels")
    parser.add_argument("--line-width", type=int, default=2,
                        help="2D center-to-face link width in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_views <= 0:
        raise ValueError("--num-views must be positive")
    if args.point_radius <= 0 or args.line_width <= 0:
        raise ValueError("--point-radius and --line-width must be positive")
    visualize_scene(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    main()
