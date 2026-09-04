#!/usr/bin/env python3
"""Visualize aligned ScanNet GT and VGGT-Omega reconstructed point clouds.

VGGT is also written in its original local coordinates. Its comparison cloud
is scaled from the scene extent, then mapped through the first camera into
ScanNet's axis-aligned GT coordinates; no ICP is applied.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import mmengine
import numpy as np
import open3d as o3d
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera


AXIS_ALIGNED_OUTPUT_MARKER = ".axis_aligned_vggt_scaled_v3"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write ScanNet GT and VGGT-Omega point clouds for ScanNet scenes.")
    parser.add_argument(
        "--scene-id", help="Process only this scene. Omit to process every scene.")
    parser.add_argument(
        "--data-root", default="/lv_workdir/data/ScanNet_processed")
    parser.add_argument(
        "--ann-file", default="scannet_infos_train_pts.pkl")
    parser.add_argument(
        "--checkpoint", default="/lv_workdir/data/pretrain/vggt_omega_1b_512.pt")
    parser.add_argument("--output-dir", default="vis/vggt_pointcloud")
    parser.add_argument("--num-views", type=int, default=100)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--max-render-points", type=int, default=50000)
    parser.add_argument("--max-depth", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many selected scenes.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate scenes whose point-cloud outputs and comparison image exist.")
    return parser.parse_args()


def find_scene(data_list, scene_id):
    for item in data_list:
        if Path(item["lidar_points"]["lidar_path"]).stem == scene_id:
            return item
    raise KeyError(f"Scene {scene_id!r} was not found in the annotation file")


def select_views(item, data_root, num_views):
    candidates = [
        index for index, relative_path in enumerate(item["img_paths"])
        if (data_root / relative_path).is_file()
    ]
    if not candidates:
        raise FileNotFoundError("The scene has no readable RGB frames")
    count = min(num_views, len(candidates))
    positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
    return [candidates[position] for position in positions]


def load_images(item, data_root, indices, width, height):
    images = []
    for index in indices:
        path = data_root / item["img_paths"][index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale = min(width / image.shape[1], height / image.shape[0])
        resized = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((height, width, 3), dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        images.append(padded)
    return np.stack(images)


def load_gt_points(item, data_root):
    point_path = data_root / "points" / item["lidar_points"]["lidar_path"]
    point_dim = item["lidar_points"].get("num_pts_feats", 6)
    raw = np.fromfile(point_path, dtype=np.float32)
    if raw.size % point_dim:
        raise ValueError(f"Unexpected point-cloud shape in {point_path}")
    raw = raw.reshape(-1, point_dim)
    points = raw[:, :3]
    colors = raw[:, 3:6] if point_dim >= 6 else np.full_like(points, 0.65)
    if colors.max(initial=0.0) > 1.0:
        colors = colors / 255.0
    return points, np.clip(colors, 0.0, 1.0)


def align_gt_points(points, axis_align_matrix):
    """Match ScanNet's aligned GT box coordinate system."""
    matrix = np.asarray(axis_align_matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"Expected axis_align_matrix with shape (4, 4), got {matrix.shape}")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def align_vggt_points(points, axis_align_matrix, first_frame_c2w,
                      predicted_first_w2c):
    """Map VGGT's first-frame world coordinates into ScanNet's aligned world."""
    matrices = {
        "axis_align_matrix": axis_align_matrix,
        "first_frame_c2w": first_frame_c2w,
        "predicted_first_w2c": predicted_first_w2c,
    }
    for name, matrix in matrices.items():
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape == (3, 4):
            matrix = np.concatenate(
                [matrix, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)],
                axis=0)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected {name} with shape (4, 4), got {matrix.shape}")
        matrices[name] = matrix
    transform = (matrices["axis_align_matrix"] @ matrices["first_frame_c2w"]
                 @ matrices["predicted_first_w2c"])
    return points @ transform[:3, :3].T + transform[:3, 3]


def point_cloud_range(points):
    """Return a robust scene extent without letting sparse outliers dominate."""
    lower, upper = np.quantile(points, [0.01, 0.99], axis=0)
    span = upper - lower
    return lower, upper, span, float(np.linalg.norm(span))


def estimate_scene_scale(gt_points, vggt_range):
    """Estimate an isotropic VGGT-to-GT scale from whole-scene extents."""
    gt_range = point_cloud_range(gt_points)
    if vggt_range[3] <= 1e-6:
        raise ValueError("VGGT point-cloud range is too small to estimate scale")
    return gt_range[3] / vggt_range[3], gt_range, vggt_range


def load_vggt(checkpoint_path, device):
    model = VGGTOmega(
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_object_queries=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval().to(device)


@torch.no_grad()
def reconstruct_vggt(model, images, point_stride, max_depth, max_points, device):
    image_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
    image_tensor = image_tensor.unsqueeze(0).float().div_(255.0).to(device)
    predictions = model(image_tensor)
    depth = predictions["depth"][0, ..., 0].float()
    pose = predictions["pose_enc"].float()
    extrinsics, intrinsics = encoding_to_camera(
        pose, image_tensor.shape[-2:])

    views, height, width = depth.shape
    u, v = torch.meshgrid(
        torch.arange(0, width, point_stride, device=device, dtype=torch.float32),
        torch.arange(0, height, point_stride, device=device, dtype=torch.float32),
        indexing="xy")
    sampled_depth = depth[:, ::point_stride, ::point_stride]
    sampled_colors = image_tensor[0].permute(0, 2, 3, 1)[
        :, ::point_stride, ::point_stride]
    points, colors = [], []
    for view_index in range(views):
        d = sampled_depth[view_index]
        k = intrinsics[0, view_index]
        camera_points = torch.stack([
            (u - k[0, 2]) * d / k[0, 0].clamp_min(1e-6),
            (v - k[1, 2]) * d / k[1, 1].clamp_min(1e-6),
            d,
        ], dim=-1)
        extrinsic = extrinsics[0, view_index]
        world = torch.einsum(
            "ij,hwj->hwi", extrinsic[:3, :3].transpose(0, 1),
            camera_points - extrinsic[:3, 3])
        valid = torch.isfinite(world).all(dim=-1)
        valid &= torch.isfinite(d) & (d > 1e-4) & (d < max_depth)
        points.append(world[valid])
        colors.append(sampled_colors[view_index][valid])
    points = torch.cat(points)
    colors = torch.cat(colors)
    full_vggt_range = point_cloud_range(points.cpu().numpy())
    if points.shape[0] > max_points:
        keep = torch.randperm(points.shape[0], device=device)[:max_points]
        points, colors = points[keep], colors[keep]
    return (points.cpu().numpy(), colors.cpu().numpy(),
            extrinsics[0, 0].cpu().numpy(), full_vggt_range)


def write_ply(path, points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"Could not write point cloud: {path}")


def sample_for_render(points, colors, max_points):
    if len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(0)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices], colors[indices]


def point_bounds(points):
    low, high = np.quantile(points, [0.01, 0.99], axis=0)
    span = max(float((high - low).max()), 1e-3)
    margin = span * 0.08
    return low - margin, high + margin


def add_coordinate_axes(axis, lower, upper):
    origin = np.zeros(3, dtype=np.float32)
    axis_length = max(float((upper - lower).max()) * 0.18, 0.1)
    for index, (color, name) in enumerate((("#d62728", "X"),
                                            ("#2ca02c", "Y"),
                                            ("#1f77b4", "Z"))):
        endpoint = origin.copy()
        endpoint[index] = axis_length
        axis.quiver(*origin, *(endpoint - origin), color=color,
                    arrow_length_ratio=0.12, linewidth=2.0)
        axis.text(*endpoint, name, color=color, fontsize=10, weight="bold")


def configure_3d_axis(axis, lower, upper, title):
    axis.set_title(title, pad=14)
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_box_aspect(upper - lower)
    axis.view_init(elev=22, azim=-56)
    add_coordinate_axes(axis, lower, upper)


def visualize_point_clouds(path, gt_points, gt_colors, vggt_points, vggt_colors,
                           vggt_aligned_points, max_points):
    gt_points, gt_colors = sample_for_render(gt_points, gt_colors, max_points)
    if len(vggt_points) > max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(vggt_points), size=max_points, replace=False)
        vggt_points = vggt_points[indices]
        vggt_colors = vggt_colors[indices]
        vggt_aligned_points = vggt_aligned_points[indices]
    gt_lower, gt_upper = point_bounds(gt_points)
    vggt_lower, vggt_upper = point_bounds(vggt_points)
    overlay_lower, overlay_upper = point_bounds(
        np.concatenate([gt_points, vggt_aligned_points], axis=0))

    figure = plt.figure(figsize=(18, 6.5), constrained_layout=True)
    gt_axis = figure.add_subplot(1, 3, 1, projection="3d")
    gt_axis.scatter(*gt_points.T, c=gt_colors, s=0.45, alpha=0.7,
                    linewidths=0, rasterized=True)
    configure_3d_axis(
        gt_axis, gt_lower, gt_upper, "GT point cloud in GT coordinates")

    vggt_axis = figure.add_subplot(1, 3, 2, projection="3d")
    vggt_axis.scatter(*vggt_points.T, c=vggt_colors, s=0.45, alpha=0.7,
                      linewidths=0, rasterized=True)
    configure_3d_axis(
        vggt_axis, vggt_lower, vggt_upper,
        "VGGT point cloud in VGGT coordinates")

    overlay_axis = figure.add_subplot(1, 3, 3, projection="3d")
    overlay_axis.scatter(*gt_points.T, c="#2878d4", s=0.5, alpha=0.5,
                         linewidths=0, rasterized=True)
    overlay_axis.scatter(*vggt_aligned_points.T, c="#e76f23", s=0.5, alpha=0.75,
                         linewidths=0, rasterized=True)
    configure_3d_axis(
        overlay_axis, overlay_lower, overlay_upper,
        "GT and VGGT point clouds in GT coordinates")
    overlay_axis.legend(handles=[
        Line2D([0], [0], marker="o", color="w", label="GT", markerfacecolor="#2878d4", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="VGGT axis-aligned", markerfacecolor="#e76f23", markersize=8),
    ], loc="upper right")
    figure.suptitle("VGGT is scaled, then mapped through the first camera into GT coordinates")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def output_signature(args):
    return {
        "num_views": args.num_views,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "point_stride": args.point_stride,
        "max_depth": args.max_depth,
        "max_points": args.max_points,
        "checkpoint": str(args.checkpoint),
    }


def output_is_complete(output_dir, args):
    return all((output_dir / name).is_file() for name in (
        AXIS_ALIGNED_OUTPUT_MARKER,
        "gt_in_gt_coordinates.ply",
        "vggt_in_vggt_coordinates.ply",
        "vggt_in_gt_axis_aligned_coordinates.ply",
        "gt_and_vggt_in_gt_coordinates_axis_aligned.ply",
        "comparison.png",
    )) and (output_dir / AXIS_ALIGNED_OUTPUT_MARKER).read_text() == json.dumps(
        output_signature(args), sort_keys=True)


def process_scene(item, scene_id, model, args, data_root, output_root):
    output_dir = output_root / scene_id
    if not args.overwrite and output_is_complete(output_dir, args):
        print(f"[{scene_id}] skipped: output already exists")
        return "skipped"
    output_dir.mkdir(parents=True, exist_ok=True)
    view_indices = select_views(item, data_root, args.num_views)
    images = load_images(
        item, data_root, view_indices, args.image_width, args.image_height)
    gt_points, gt_colors = load_gt_points(item, data_root)
    gt_points = align_gt_points(gt_points, item["axis_align_matrix"])
    vggt_points, vggt_colors, predicted_first_w2c, vggt_range = reconstruct_vggt(
        model, images, args.point_stride, args.max_depth, args.max_points,
        torch.device(args.device))
    scene_scale, gt_range, vggt_range = estimate_scene_scale(
        gt_points, vggt_range)
    vggt_aligned_points = align_vggt_points(
        vggt_points * scene_scale, item["axis_align_matrix"],
        item["lidar2cam"][view_indices[0]], predicted_first_w2c)

    write_ply(output_dir / "gt_in_gt_coordinates.ply", gt_points, gt_colors)
    write_ply(output_dir / "vggt_in_vggt_coordinates.ply", vggt_points, vggt_colors)
    write_ply(
        output_dir / "vggt_in_gt_axis_aligned_coordinates.ply",
        vggt_aligned_points, vggt_colors)
    overlay_points = np.concatenate([gt_points, vggt_aligned_points], axis=0)
    overlay_colors = np.concatenate([
        np.broadcast_to(np.array([0.16, 0.47, 0.83]), gt_points.shape),
        np.broadcast_to(np.array([0.91, 0.44, 0.14]), vggt_points.shape),
    ], axis=0)
    write_ply(
        output_dir / "gt_and_vggt_in_gt_coordinates_axis_aligned.ply",
        overlay_points, overlay_colors)
    visualize_point_clouds(
        output_dir / "comparison.png", gt_points, gt_colors,
        vggt_points, vggt_colors, vggt_aligned_points,
        args.max_render_points)
    (output_dir / AXIS_ALIGNED_OUTPUT_MARKER).write_text(
        json.dumps(output_signature(args), sort_keys=True))
    print(f"[{scene_id}] selected frames: {view_indices}")
    print(
        f"[{scene_id}] GT points: {len(gt_points)}, "
        f"range: {gt_range[0]} -> {gt_range[1]}, "
        f"span={gt_range[2]}, diagonal={gt_range[3]:.4f}")
    print(
        f"[{scene_id}] VGGT reconstructed points: {len(vggt_points)}, "
        f"range: {vggt_range[0]} -> {vggt_range[1]}, "
        f"span={vggt_range[2]}, diagonal={vggt_range[3]:.4f}")
    print(f"[{scene_id}] VGGT-to-GT scene scale: {scene_scale:.6f}")
    print(
        f"[{scene_id}] axis-aligned VGGT bounds: "
        f"{vggt_aligned_points.min(0)} -> {vggt_aligned_points.max(0)}")
    print(f"[{scene_id}] wrote: {output_dir}")
    return "processed"


def main():
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    annotation = mmengine.load(data_root / args.ann_file)
    data_list = annotation["data_list"]
    if args.scene_id:
        scenes = [(args.scene_id, find_scene(data_list, args.scene_id))]
    else:
        scenes = [
            (Path(item["lidar_points"]["lidar_path"]).stem, item)
            for item in data_list]
    if args.limit is not None:
        scenes = scenes[:args.limit]
    if not scenes:
        raise ValueError("No scenes selected")

    if not args.overwrite:
        pending_scenes = []
        skipped = 0
        for scene_id, item in scenes:
            if output_is_complete(output_root / scene_id, args):
                print(f"[{scene_id}] skipped: output already exists")
                skipped += 1
            else:
                pending_scenes.append((scene_id, item))
        scenes = pending_scenes
    else:
        skipped = 0
    if not scenes:
        print(f"Finished: processed=0, skipped={skipped}, failed=0")
        return

    device = torch.device(args.device)
    model = load_vggt(args.checkpoint, device)
    processed = failed = 0
    for index, (scene_id, item) in enumerate(scenes, start=1):
        print(f"[{index}/{len(scenes)}] {scene_id}")
        try:
            outcome = process_scene(
                item, scene_id, model, args, data_root, output_root)
            if outcome == "processed":
                processed += 1
            else:
                skipped += 1
        except Exception as error:
            failed += 1
            print(f"[{scene_id}] failed: {type(error).__name__}: {error}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(
        f"Finished {len(scenes)} scenes: processed={processed}, "
        f"skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
