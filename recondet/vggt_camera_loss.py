from collections.abc import Sequence

import torch

from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding


VGGT_CAMERA_LOSS_DEFAULTS = dict(
    weight=5.0,
    loss_type='l1',
    gamma=0.6,
    weight_trans=1.0,
    weight_rot=1.0,
    weight_focal=0.5,
    min_valid_points=100,
)


def _sanitize_loss(loss):
    loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
    return loss.clamp(min=-100, max=100)


def _camera_loss_single(prediction, target, loss_type):
    if loss_type == 'l1':
        translation = (prediction[..., :3] - target[..., :3]).abs()
        rotation = (prediction[..., 3:7] - target[..., 3:7]).abs()
        focal = (prediction[..., 7:] - target[..., 7:]).abs()
    elif loss_type == 'l2':
        translation = (
            prediction[..., :3] - target[..., :3]
        ).norm(dim=-1, keepdim=True)
        rotation = (
            prediction[..., 3:7] - target[..., 3:7]
        ).norm(dim=-1)
        focal = (
            prediction[..., 7:] - target[..., 7:]
        ).norm(dim=-1)
    else:
        raise ValueError(f'Unknown camera loss type: {loss_type}')

    translation = _sanitize_loss(translation).clamp(max=100).mean()
    rotation = _sanitize_loss(rotation).mean()
    focal = _sanitize_loss(focal).mean()
    return translation, rotation, focal


def compute_vggt_camera_loss(
        pose_encodings,
        gt_extrinsics,
        gt_intrinsics,
        point_masks,
        image_hw,
        weight=5.0,
        loss_type='l1',
        gamma=0.6,
        weight_trans=1.0,
        weight_rot=1.0,
        weight_focal=0.5,
        min_valid_points=100):
    """Apply VGGT's camera objective to one or more prediction stages."""
    if isinstance(pose_encodings, torch.Tensor):
        pose_encodings = [pose_encodings]
    elif isinstance(pose_encodings, Sequence):
        pose_encodings = list(pose_encodings)
    else:
        raise TypeError('pose_encodings must be a Tensor or a sequence')
    if not pose_encodings:
        raise ValueError('pose_encodings cannot be empty')
    if point_masks.ndim != 4:
        raise ValueError('point_masks must have shape [B, V, H, W]')

    prediction = pose_encodings[0]
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics.to(device=prediction.device, dtype=torch.float32),
        gt_intrinsics.to(device=prediction.device, dtype=torch.float32),
        image_hw)
    valid_scene_mask = (
        point_masks[:, 0].sum(dim=(-1, -2)) > min_valid_points)

    total_translation = prediction.new_zeros(())
    total_rotation = prediction.new_zeros(())
    total_focal = prediction.new_zeros(())
    num_stages = len(pose_encodings)
    for stage_index, stage_prediction in enumerate(pose_encodings):
        if stage_prediction.shape != gt_pose_encoding.shape:
            raise ValueError(
                'Camera prediction and ground truth shapes differ: '
                f'{tuple(stage_prediction.shape)} vs '
                f'{tuple(gt_pose_encoding.shape)}')
        if valid_scene_mask.any():
            translation, rotation, focal = _camera_loss_single(
                stage_prediction[valid_scene_mask].clone(),
                gt_pose_encoding[valid_scene_mask].clone(),
                loss_type)
        else:
            zero = (stage_prediction * 0).mean()
            translation = rotation = focal = zero
        stage_weight = gamma ** (num_stages - stage_index - 1)
        total_translation += stage_weight * translation
        total_rotation += stage_weight * rotation
        total_focal += stage_weight * focal

    translation = total_translation / num_stages
    rotation = total_rotation / num_stages
    focal = total_focal / num_stages
    camera_loss = weight * (
        weight_trans * translation
        + weight_rot * rotation
        + weight_focal * focal)
    return {
        'vggt_loss_camera': camera_loss,
        'camera_translation_error': translation.detach(),
        'camera_rotation_error': rotation.detach(),
        'camera_fov_error': focal.detach(),
    }
