#!/usr/bin/env python3
"""Analyze VGGT-Omega scale and align one reconstruction to ScanNet GT."""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_DATA_ROOT = Path('/root/shared-nvme/data/ScanNet_processed_v2')
DEFAULT_CHECKPOINT = Path(
    '/root/shared-nvme/data/vggt-omega/vggt_omega_1b_512.pt')
GT_COLOR = np.array([41, 120, 212], dtype=np.uint8)
VGGT_COLOR = np.array([232, 112, 36], dtype=np.uint8)


def as_homogeneous(matrix, name):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.concatenate(
            [matrix, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    if matrix.shape != (4, 4):
        raise ValueError(f'{name} must have shape [3,4] or [4,4]')
    if not np.isfinite(matrix).all():
        raise ValueError(f'{name} contains non-finite values')
    return matrix


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    transform = as_homogeneous(transform, 'transform')
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape [N,3]')
    return points @ transform[:3, :3].T + transform[:3, 3]


def finite_points(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape [N,3]')
    return points[np.isfinite(points).all(axis=1)]


def mean_distance_from_origin(points):
    points = finite_points(points)
    if len(points) == 0:
        raise ValueError('Point cloud has no finite points')
    distances = np.linalg.norm(points, axis=1)
    distances = distances[np.isfinite(distances)]
    if len(distances) == 0:
        raise ValueError('Point cloud has no finite points')
    mean_distance = float(distances.mean())
    if not np.isfinite(mean_distance) or mean_distance <= 1e-8:
        raise ValueError(f'Invalid mean point distance: {mean_distance}')
    return mean_distance


def scene_id_from_item(item):
    image_paths = item.get('img_paths')
    if not image_paths:
        raise ValueError('Annotation item has no img_paths')
    scene_id = Path(image_paths[0]).parent.name
    if not scene_id:
        raise ValueError(f'Cannot derive scene id from {image_paths[0]!r}')
    return scene_id


def scale_and_align_vggt_points(vggt_world_points, predicted_first_w2c,
                                gt_aligned_points, gt_first_c2w_aligned):
    """Scale around the first camera, then map into aligned ScanNet world."""
    predicted_first_w2c = as_homogeneous(
        predicted_first_w2c, 'predicted_first_w2c')
    gt_first_c2w_aligned = as_homogeneous(
        gt_first_c2w_aligned, 'gt_first_c2w_aligned')

    vggt_first_points = transform_points(
        finite_points(vggt_world_points), predicted_first_w2c)
    gt_first_points = transform_points(
        finite_points(gt_aligned_points),
        np.linalg.inv(gt_first_c2w_aligned))
    vggt_mean = mean_distance_from_origin(vggt_first_points)
    gt_mean = mean_distance_from_origin(gt_first_points)
    scale_ratio = gt_mean / vggt_mean

    scaled_vggt_first = vggt_first_points * scale_ratio
    aligned_vggt = transform_points(
        scaled_vggt_first, gt_first_c2w_aligned)
    return aligned_vggt.astype(np.float32), {
        'vggt_mean_distance': vggt_mean,
        'gt_mean_distance': gt_mean,
        'scale_ratio': scale_ratio,
    }


def load_annotation(path):
    with path.open('rb') as file:
        annotation = pickle.load(file)
    if not isinstance(annotation, dict) or 'data_list' not in annotation:
        raise ValueError('Annotation must contain a data_list')
    return annotation['data_list']


def find_scene(data_list, scene_id):
    for item in data_list:
        if scene_id_from_item(item) == scene_id:
            return item
    raise KeyError(f'Scene {scene_id!r} was not found in the annotation')


def select_view_indices(item, data_root, num_views):
    candidates = [
        index for index, relative_path in enumerate(item['img_paths'])
        if (data_root / relative_path).is_file()
    ]
    if not candidates:
        raise FileNotFoundError('Scene has no readable RGB frames')
    count = min(num_views, len(candidates))
    positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
    return [candidates[position] for position in positions]


def load_images(item, data_root, indices, long_side, pad_divisor):
    resized_images = []
    for index in indices:
        path = data_root / item['img_paths'][index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Could not read image: {path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale = long_side / max(image.shape[:2])
        width = max(1, int(round(image.shape[1] * scale)))
        height = max(1, int(round(image.shape[0] * scale)))
        resized_images.append(cv2.resize(
            image, (width, height), interpolation=cv2.INTER_CUBIC))

    max_height = max(image.shape[0] for image in resized_images)
    max_width = max(image.shape[1] for image in resized_images)
    padded_height = int(np.ceil(max_height / pad_divisor) * pad_divisor)
    padded_width = int(np.ceil(max_width / pad_divisor) * pad_divisor)
    images = np.zeros(
        (len(resized_images), padded_height, padded_width, 3), dtype=np.uint8)
    for index, image in enumerate(resized_images):
        images[index, :image.shape[0], :image.shape[1]] = image
    return images


def load_axis_aligned_gt(item, data_root, scene_id, point_dim):
    point_path = data_root / 'points' / f'{scene_id}.bin'
    raw = np.fromfile(point_path, dtype=np.float32)
    if raw.size == 0 or raw.size % point_dim:
        raise ValueError(f'Unexpected point cloud shape in {point_path}')
    points = raw.reshape(-1, point_dim)[:, :3]
    axis_align = as_homogeneous(item['axis_align_matrix'], 'axis_align_matrix')
    return transform_points(points, axis_align).astype(np.float32), axis_align


def load_vggt(checkpoint, device):
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega()
    state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


@torch.inference_mode()
def reconstruct_vggt(model, images, point_stride, min_depth, max_depth,
                     device):
    from vggt_omega.utils.geometry import (
        unproject_depth_map_to_point_map_torch)
    from vggt_omega.utils.pose_enc import encoding_to_camera

    image_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
    image_tensor = image_tensor.unsqueeze(0).float().div(255.0).to(device)
    predictions = model(image_tensor)
    depth = predictions['depth'].float()[..., 0]
    extrinsics, intrinsics = encoding_to_camera(
        predictions['pose_enc'].float(), image_tensor.shape[-2:])
    world_points = unproject_depth_map_to_point_map_torch(
        depth, extrinsics, intrinsics)

    sampled_depth = depth[0, :, ::point_stride, ::point_stride]
    sampled_points = world_points[0, :, ::point_stride, ::point_stride]
    valid = torch.isfinite(sampled_points).all(dim=-1)
    valid &= torch.isfinite(sampled_depth)
    valid &= sampled_depth > min_depth
    valid &= sampled_depth < max_depth
    points = sampled_points[valid]
    if points.numel() == 0:
        raise ValueError('VGGT reconstruction has no valid points')
    return (
        points.cpu().numpy().astype(np.float32),
        extrinsics[0, 0].cpu().numpy().astype(np.float32),
        tuple(image_tensor.shape[-2:]),
    )


def sample_points(points, maximum, seed):
    if maximum is None or len(points) <= maximum:
        return points
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(points), maximum, replace=False)
    return points[indices]


def write_colored_ply(path, points, colors):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points and colors must both have shape [N,3]')
    vertices = np.empty(
        len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'], vertices['y'], vertices['z'] = points.T
    vertices['red'], vertices['green'], vertices['blue'] = colors.T
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {len(vertices)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as file:
        file.write(header.encode('ascii'))
        vertices.tofile(file)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize one VGGT-Omega ScanNet reconstruction')
    parser.add_argument('--scene-id', required=True)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--ann-file', type=Path,
                        default=Path('scannet_infos_train_mvod.pkl'))
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--num-views', type=int, default=50)
    parser.add_argument('--image-long-side', type=int, default=448)
    parser.add_argument('--pad-divisor', type=int, default=16)
    parser.add_argument('--point-stride', type=int, default=4)
    parser.add_argument('--point-dim', type=int, default=6)
    parser.add_argument('--min-depth', type=float, default=1e-4)
    parser.add_argument('--max-depth', type=float, default=30.0)
    parser.add_argument('--max-vggt-points', type=int, default=250000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def validate_args(args):
    positive = ('num_views', 'image_long_side', 'pad_divisor',
                'point_stride', 'point_dim', 'max_vggt_points')
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        raise ValueError('Depth range must satisfy 0 <= min-depth < max-depth')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')


def main():
    args = parse_args()
    validate_args(args)
    annotation_path = (
        args.ann_file if args.ann_file.is_absolute()
        else args.data_root / args.ann_file)
    item = find_scene(load_annotation(annotation_path), args.scene_id)
    view_indices = select_view_indices(item, args.data_root, args.num_views)
    images = load_images(
        item, args.data_root, view_indices,
        args.image_long_side, args.pad_divisor)
    gt_aligned, axis_align = load_axis_aligned_gt(
        item, args.data_root, args.scene_id, args.point_dim)

    device = torch.device(args.device)
    model = load_vggt(args.checkpoint, device)
    vggt_world, predicted_first_w2c, image_shape = reconstruct_vggt(
        model, images, args.point_stride, args.min_depth,
        args.max_depth, device)

    gt_first_c2w = as_homogeneous(
        item['lidar2cam'][view_indices[0]], 'gt_first_c2w')
    gt_first_c2w_aligned = axis_align @ gt_first_c2w
    vggt_aligned, stats = scale_and_align_vggt_points(
        vggt_world, predicted_first_w2c,
        gt_aligned, gt_first_c2w_aligned)

    vggt_output = sample_points(
        vggt_aligned, args.max_vggt_points, args.seed)
    overlay_points = np.concatenate([gt_aligned, vggt_output], axis=0)
    overlay_colors = np.concatenate([
        np.broadcast_to(GT_COLOR, gt_aligned.shape),
        np.broadcast_to(VGGT_COLOR, vggt_output.shape),
    ], axis=0)
    output = args.output or (
        Path('outputs') / 'vggt_scale_analysis' /
        f'{args.scene_id}_gt_vggt_aligned.ply')
    write_colored_ply(output, overlay_points, overlay_colors)

    print(f'scene_id: {args.scene_id}')
    print(f'first_view_index: {view_indices[0]}')
    print(f'selected_views: {len(view_indices)}')
    print(f'input_shape: {image_shape[0]}x{image_shape[1]}')
    print(f'gt_points: {len(gt_aligned)}')
    print(f'vggt_valid_points: {len(vggt_world)}')
    print(f'vggt_output_points: {len(vggt_output)}')
    print(f'vggt_mean_distance: {stats["vggt_mean_distance"]:.8f}')
    print(f'gt_mean_distance: {stats["gt_mean_distance"]:.8f}')
    print(f'scale_ratio_gt_over_vggt: {stats["scale_ratio"]:.8f}')
    print(f'output_ply: {output.resolve()}')


if __name__ == '__main__':
    main()
