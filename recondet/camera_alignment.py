from pathlib import Path

import numpy as np
import torch


def _batch_matrix(value, reference, batch_size, name):
    if isinstance(value, torch.Tensor):
        matrix = value
    elif isinstance(value, np.ndarray):
        matrix = torch.from_numpy(value)
    else:
        matrix = torch.stack([
            item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
            for item in value
        ])
    matrix = matrix.to(device=reference.device, dtype=torch.float32)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    if matrix.shape != (batch_size, 4, 4):
        raise ValueError(
            f'{name} must have shape [B, 4, 4], got {tuple(matrix.shape)}')
    return matrix


def _homogeneous_extrinsics(extrinsics):
    if extrinsics.ndim != 4:
        raise ValueError(
            'VGGT extrinsics must have shape [B, V, 3, 4] or [B, V, 4, 4]')
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics.float()
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(
            'VGGT extrinsics must have shape [B, V, 3, 4] or [B, V, 4, 4]')
    homogeneous = extrinsics.new_zeros(*extrinsics.shape[:-2], 4, 4)
    homogeneous[..., :3, :] = extrinsics
    homogeneous[..., 3, 3] = 1
    return homogeneous.float()


def _gt_inverse_components(reference, first_frame_pose, axis_align_matrix,
                           scene_scale):
    batch_size = reference.shape[0]
    first_frame_pose = _batch_matrix(
        first_frame_pose, reference, batch_size, 'first_frame_pose')
    axis_align_matrix = _batch_matrix(
        axis_align_matrix, reference, batch_size, 'axis_align_matrix')
    scene_scale = torch.as_tensor(
        scene_scale, device=reference.device, dtype=torch.float32).reshape(-1)
    if scene_scale.shape != (batch_size,):
        raise ValueError('scene_scale must contain one value per batch item')
    if not torch.isfinite(scene_scale).all() or (scene_scale <= 0).any():
        raise ValueError('scene_scale must contain finite positive values')

    scale_matrix = torch.eye(
        4, device=reference.device, dtype=torch.float32).repeat(
            batch_size, 1, 1)
    scale_matrix[:, :3, :3] *= scene_scale[:, None, None]
    first_c2w_aligned = torch.bmm(axis_align_matrix, first_frame_pose)
    normalized_to_aligned = torch.bmm(first_c2w_aligned, scale_matrix)
    return scale_matrix, normalized_to_aligned, scene_scale


@torch.no_grad()
def denormalize_vggt_gt_points(points, first_frame_pose, axis_align_matrix,
                               scene_scale):
    """Invert BuildVGGTGroundTruth's first-camera point normalization."""
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError('points must have shape [B, V, Q, 3]')
    with torch.autocast(device_type=points.device.type, enabled=False):
        _, normalized_to_aligned, _ = _gt_inverse_components(
            points, first_frame_pose, axis_align_matrix, scene_scale)
        aligned_points = torch.einsum(
            'bij,bvqj->bvqi', normalized_to_aligned[:, :3, :3],
            points.float())
        aligned_points = aligned_points + normalized_to_aligned[
            :, None, None, :3, 3]
    return aligned_points


@torch.no_grad()
def denormalize_vggt_gt_cameras(extrinsics, first_frame_pose,
                                axis_align_matrix, scene_scale):
    """Map normalized VGGT W2C cameras into axis-aligned metric space."""
    extrinsics_h = _homogeneous_extrinsics(extrinsics)
    with torch.autocast(device_type=extrinsics.device.type, enabled=False):
        scale_matrix, normalized_to_aligned, _ = _gt_inverse_components(
            extrinsics_h, first_frame_pose, axis_align_matrix, scene_scale)
        metric_camera_extrinsics = torch.matmul(
            scale_matrix[:, None], extrinsics_h)
        aligned_extrinsics = torch.matmul(
            metric_camera_extrinsics,
            torch.linalg.inv(normalized_to_aligned)[:, None])
    if not torch.isfinite(aligned_extrinsics).all():
        raise FloatingPointError('Aligned VGGT extrinsics contain non-finite values')
    return aligned_extrinsics[..., :3, :]


def _alignment_components(extrinsics, first_frame_pose, axis_align_matrix,
                          scene_scale):
    extrinsics_h = _homogeneous_extrinsics(extrinsics)
    batch_size = extrinsics_h.shape[0]
    first_frame_pose = _batch_matrix(
        first_frame_pose, extrinsics_h, batch_size, 'first_frame_pose')
    axis_align_matrix = _batch_matrix(
        axis_align_matrix, extrinsics_h, batch_size, 'axis_align_matrix')
    scene_scale = torch.as_tensor(
        scene_scale, device=extrinsics_h.device, dtype=torch.float32).reshape(-1)
    if scene_scale.shape != (batch_size,):
        raise ValueError('scene_scale must contain one value per batch item')
    if not torch.isfinite(scene_scale).all() or (scene_scale <= 0).any():
        raise ValueError('scene_scale must contain finite positive values')

    alignment = torch.bmm(
        axis_align_matrix, torch.bmm(first_frame_pose, extrinsics_h[:, 0]))
    scale_matrix = torch.eye(
        4, device=extrinsics_h.device, dtype=torch.float32).repeat(
            batch_size, 1, 1)
    scale_matrix[:, :3, :3] *= scene_scale[:, None, None]
    return extrinsics_h, alignment, scale_matrix, scene_scale


@torch.no_grad()
def align_vggt_cameras(extrinsics, first_frame_pose, axis_align_matrix,
                       scene_scale):
    with torch.autocast(device_type=extrinsics.device.type, enabled=False):
        extrinsics_h, alignment, scale_matrix, _ = _alignment_components(
            extrinsics, first_frame_pose, axis_align_matrix, scene_scale)
        vggt_to_aligned = torch.bmm(alignment, scale_matrix)
        scaled_camera_extrinsics = torch.matmul(
            scale_matrix[:, None], extrinsics_h)
        aligned_extrinsics = torch.matmul(
            scaled_camera_extrinsics,
            torch.linalg.inv(vggt_to_aligned)[:, None])

    if not torch.isfinite(aligned_extrinsics).all():
        raise FloatingPointError('Aligned VGGT extrinsics contain non-finite values')
    return aligned_extrinsics[..., :3, :]


@torch.no_grad()
def align_vggt_reconstruction(point_map, extrinsics, first_frame_pose,
                              axis_align_matrix, scene_scale):
    if point_map.ndim != 5 or point_map.shape[-1] != 3:
        raise ValueError('VGGT point map must have shape [B, V, H, W, 3]')
    batch_size, num_views = point_map.shape[:2]
    extrinsics_h = _homogeneous_extrinsics(extrinsics)
    if extrinsics_h.shape[:2] != (batch_size, num_views):
        raise ValueError('VGGT point map and extrinsics must share B and V')

    with torch.autocast(device_type=point_map.device.type, enabled=False):
        _, alignment, _, scene_scale = _alignment_components(
            extrinsics_h, first_frame_pose, axis_align_matrix, scene_scale)

        scaled_points = point_map.float() * scene_scale[:, None, None, None, None]
        aligned_points = torch.einsum(
            'bij,bvhwj->bvhwi', alignment[:, :3, :3], scaled_points)
        aligned_points = aligned_points + alignment[
            :, None, None, None, :3, 3]

    aligned_extrinsics = align_vggt_cameras(
        extrinsics_h, first_frame_pose, axis_align_matrix, scene_scale)
    return aligned_points, aligned_extrinsics


@torch.no_grad()
def align_vggt_query_points(points, extrinsics, first_frame_pose,
                            axis_align_matrix, scene_scale):
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError('points must have shape [B, V, Q, 3]')
    if extrinsics.shape[:2] != points.shape[:2]:
        raise ValueError('query points and extrinsics must share B and V')

    with torch.autocast(device_type=points.device.type, enabled=False):
        _, alignment, _, scene_scale = _alignment_components(
            extrinsics, first_frame_pose, axis_align_matrix, scene_scale)
        scaled_points = points.float() * scene_scale[:, None, None, None]
        aligned_points = torch.einsum(
            'bij,bvqj->bvqi', alignment[:, :3, :3], scaled_points)
        aligned_points = aligned_points + alignment[:, None, None, :3, 3]
    return aligned_points


@torch.no_grad()
def unproject_sampled_depth_map(depth_map, extrinsics, intrinsics, stride):
    if depth_map.ndim != 4:
        raise ValueError('depth_map must have shape [B, V, H, W]')
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError('stride must be a positive integer')
    extrinsics_h = _homogeneous_extrinsics(extrinsics)
    if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError('intrinsics must have shape [B, V, 3, 3]')
    if extrinsics_h.shape[:2] != depth_map.shape[:2]:
        raise ValueError('depth_map and extrinsics must share B and V')
    if intrinsics.shape[:2] != depth_map.shape[:2]:
        raise ValueError('depth_map and intrinsics must share B and V')

    with torch.autocast(device_type=depth_map.device.type, enabled=False):
        depth = depth_map.float()[..., ::stride, ::stride]
        height, width = depth_map.shape[-2:]
        u, v = torch.meshgrid(
            torch.arange(0, width, stride, device=depth.device,
                         dtype=torch.float32),
            torch.arange(0, height, stride, device=depth.device,
                         dtype=torch.float32),
            indexing='xy')
        intrinsics = intrinsics.float()
        fx = intrinsics[..., 0, 0, None, None]
        fy = intrinsics[..., 1, 1, None, None]
        cx = intrinsics[..., 0, 2, None, None]
        cy = intrinsics[..., 1, 2, None, None]
        camera_points = torch.stack([
            (u - cx) * depth / fx,
            (v - cy) * depth / fy,
            depth,
        ], dim=-1)

        rotation = extrinsics_h[..., :3, :3]
        translation = extrinsics_h[..., :3, 3]
        world_points = torch.einsum(
            'bvij,bvhwj->bvhwi', rotation.transpose(-1, -2),
            camera_points - translation[:, :, None, None])
    return world_points, depth


def unproject_query_depth(pixel_xy, depth, extrinsics, intrinsics):
    """Unproject per-query camera Z-depth into the native VGGT frame."""
    if pixel_xy.ndim != 3 or pixel_xy.shape[-1] != 2:
        raise ValueError('pixel_xy must have shape [N, Q, 2]')
    if depth.ndim == 2:
        depth = depth.unsqueeze(-1)
    if depth.shape != pixel_xy.shape[:2] + (1,):
        raise ValueError('depth must have shape [N, Q] or [N, Q, 1]')
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] not in (
            (3, 4), (4, 4)):
        raise ValueError(
            'extrinsics must have shape [N, 3, 4] or [N, 4, 4]')
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError('intrinsics must have shape [N, 3, 3]')
    if not (pixel_xy.shape[0] == extrinsics.shape[0] ==
            intrinsics.shape[0]):
        raise ValueError('queries and cameras must share the batch dimension')

    device_type = depth.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        pixel_xy = pixel_xy.float()
        depth = depth.float()
        extrinsics = extrinsics.float()
        intrinsics = intrinsics.float()

        fx = intrinsics[:, 0, 0, None]
        fy = intrinsics[:, 1, 1, None]
        cx = intrinsics[:, 0, 2, None]
        cy = intrinsics[:, 1, 2, None]
        z = depth[..., 0]
        camera_points = torch.stack([
            (pixel_xy[..., 0] - cx) * z / fx,
            (pixel_xy[..., 1] - cy) * z / fy,
            z,
        ], dim=-1)

        rotation = extrinsics[:, :3, :3]
        translation = extrinsics[:, :3, 3]
        world_points = torch.einsum(
            'nij,nqj->nqi', rotation.transpose(-1, -2),
            camera_points - translation[:, None])

    return camera_points, world_points


def robust_scene_diagonal(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Point cloud must have shape [N, 3]')
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError('Point cloud has no finite points')
    lower, upper = np.quantile(points, [0.01, 0.99], axis=0)
    diagonal = float(np.linalg.norm(upper - lower))
    if not np.isfinite(diagonal) or diagonal <= 1e-6:
        raise ValueError(f'Point-cloud range is too small: {diagonal}')
    return diagonal


def estimate_scene_scale(gt_points, vggt_points):
    scale = robust_scene_diagonal(gt_points) / robust_scene_diagonal(vggt_points)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f'Invalid online scene scale: {scale}')
    return float(scale)


def load_axis_aligned_points(path, num_point_features, axis_align_matrix):
    path = Path(path)
    if num_point_features < 3:
        raise ValueError('num_point_features must be at least 3')
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % num_point_features:
        raise ValueError(f'Unexpected point-cloud shape in {path}')
    points = raw.reshape(-1, num_point_features)[:, :3]
    axis_align_matrix = np.asarray(axis_align_matrix, dtype=np.float32)
    if axis_align_matrix.shape != (4, 4):
        raise ValueError('axis_align_matrix must have shape [4, 4]')
    return (points @ axis_align_matrix[:3, :3].T +
            axis_align_matrix[:3, 3])

