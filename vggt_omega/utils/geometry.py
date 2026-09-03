import torch


def depth_to_cam_coords_points_torch(
    depth_map: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_frames, height, width = depth_map.shape
    u, v = torch.meshgrid(
        torch.arange(width, device=depth_map.device),
        torch.arange(height, device=depth_map.device),
        indexing="xy",
    )
    u = u.reshape(1, 1, height, width).expand(batch_size, num_frames, -1, -1)
    v = v.reshape(1, 1, height, width).expand(batch_size, num_frames, -1, -1)

    focal_x = intrinsic[..., 0, 0].view(batch_size, num_frames, 1, 1)
    focal_y = intrinsic[..., 1, 1].view(batch_size, num_frames, 1, 1)
    center_x = intrinsic[..., 0, 2].view(batch_size, num_frames, 1, 1)
    center_y = intrinsic[..., 1, 2].view(batch_size, num_frames, 1, 1)

    x_cam = (u - center_x) * depth_map / focal_x
    y_cam = (v - center_y) * depth_map / focal_y
    return torch.stack([x_cam, y_cam, depth_map], dim=-1)


@torch.no_grad()
def closed_form_inverse_se3_torch(se3: torch.Tensor) -> torch.Tensor:
    if se3.shape[-2:] == (3, 4):
        batch_size, num_frames = se3.shape[:2]
        extended_se3 = torch.zeros(
            batch_size,
            num_frames,
            4,
            4,
            device=se3.device,
            dtype=se3.dtype,
        )
        extended_se3[..., :3, :] = se3
        extended_se3[..., 3, 3] = 1.0
        se3 = extended_se3

    rotation = se3[..., :3, :3]
    translation = se3[..., :3, 3:]
    inverse_rotation = rotation.transpose(-1, -2)
    inverse_translation = -torch.matmul(inverse_rotation, translation)

    inverse_se3 = torch.zeros_like(se3)
    inverse_se3[..., :3, :3] = inverse_rotation
    inverse_se3[..., :3, 3:4] = inverse_translation
    inverse_se3[..., 3, 3] = 1.0
    return inverse_se3


@torch.no_grad()
def unproject_depth_map_to_point_map_torch(
    depth_map: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    cam_coords = depth_to_cam_coords_points_torch(depth_map, intrinsics)
    inverse_extrinsics = closed_form_inverse_se3_torch(extrinsics)
    cam_coords_h = torch.cat([cam_coords, torch.ones_like(cam_coords[..., :1])], dim=-1)
    world_coords_h = torch.einsum("bsij,bshwj->bshwi", inverse_extrinsics, cam_coords_h)
    return world_coords_h[..., :3]
