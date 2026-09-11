from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.dense_heads.grounding_dino_head import GroundingDINOHead
from mmdet.models.dense_heads.atss_vlfusion_head import (
    convert_grounding_to_cls_scores)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import reduce_mean
from mmengine.structures import InstanceData

from recondet.camera_alignment import unproject_query_depth


CONFIDENT_QUERY_POINT_DEFAULTS = dict(
    score_thr=0.05,
    loss_weight=1.0,
    window_fraction=0.1,
    min_window_size=2,
    max_window_size=4,
    abs_depth_tolerance=0.05,
    rel_depth_tolerance=0.01,
)

def compute_adaptive_depth_window_sizes(
        bbox_preds, image_shapes, window_fraction=0.1,
        min_window_size=2, max_window_size=4):
    if bbox_preds.ndim != 3 or bbox_preds.shape[-1] != 4:
        raise ValueError('bbox_preds must have shape [N, Q, 4]')
    if image_shapes.shape != (bbox_preds.shape[0], 2):
        raise ValueError('image_shapes must have shape [N, 2]')
    if window_fraction <= 0:
        raise ValueError('window_fraction must be positive')
    if min_window_size <= 0 or max_window_size < min_window_size:
        raise ValueError('invalid depth window size range')

    image_shapes = image_shapes.to(
        device=bbox_preds.device, dtype=bbox_preds.dtype)
    box_width = bbox_preds[..., 2] * image_shapes[:, None, 1]
    box_height = bbox_preds[..., 3] * image_shapes[:, None, 0]
    short_side = torch.minimum(box_width, box_height)
    return torch.round(
        2.0 * float(window_fraction) * short_side + 1.0
    ).long().clamp(min_window_size, max_window_size)


def _stack_depth_targets(batch_data_samples, device):
    required = (
        'gt_depth_vggt', 'gt_depth_valid_mask', 'vggt_gt_scale',
        'token_positive_map')
    for sample in batch_data_samples:
        missing = [name for name in required if not hasattr(sample, name)]
        if missing:
            raise RuntimeError(
                f'Confident query point supervision requires {missing}')
    depth_maps = torch.stack([
        sample.gt_depth_vggt.to(device=device, dtype=torch.float32)
        for sample in batch_data_samples
    ])
    valid_masks = torch.stack([
        sample.gt_depth_valid_mask.to(device=device, dtype=torch.bool)
        for sample in batch_data_samples
    ])
    scales = torch.stack([
        sample.vggt_gt_scale.to(device=device, dtype=torch.float32)
        for sample in batch_data_samples
    ])
    return depth_maps, valid_masks, scales


@torch.no_grad()
def _sample_query_depth_targets(
        reference_points, bbox_preds, batch_data_samples,
        window_fraction, min_window_size, max_window_size,
        abs_depth_tolerance, rel_depth_tolerance):
    device = reference_points.device
    depth_maps, depth_valid, scales = _stack_depth_targets(
        batch_data_samples, device)
    if depth_maps.ndim != 3 or depth_valid.shape != depth_maps.shape:
        raise ValueError('GT depth maps and masks must have shape [N, H, W]')
    if len(depth_maps) != len(reference_points):
        raise ValueError('GT depth view count must match reconstruction views')
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('vggt_gt_scale must be finite and positive')

    image_shapes = reference_points.new_tensor([
        sample.metainfo['img_shape'][:2]
        for sample in batch_data_samples
    ])
    window_sizes = compute_adaptive_depth_window_sizes(
        bbox_preds, image_shapes, window_fraction,
        min_window_size, max_window_size)
    pixel_scale = torch.stack(
        [image_shapes[:, 1], image_shapes[:, 0]], dim=-1)
    centers = torch.round(
        reference_points * pixel_scale[:, None]).long()

    offset_start = -(max_window_size // 2)
    offset_end = max_window_size - max_window_size // 2
    offset_axis = torch.arange(
        offset_start, offset_end, device=device)
    offset_y, offset_x = torch.meshgrid(
        offset_axis, offset_axis, indexing='ij')
    offset_x = offset_x.reshape(1, 1, -1)
    offset_y = offset_y.reshape(1, 1, -1)
    x = centers[..., 0, None] + offset_x
    y = centers[..., 1, None] + offset_y

    left = window_sizes[..., None] // 2
    right = window_sizes[..., None] - left
    active = ((offset_x >= -left) & (offset_x < right)
              & (offset_y >= -left) & (offset_y < right))
    actual_height = image_shapes[:, None, 0, None].long()
    actual_width = image_shapes[:, None, 1, None].long()
    in_bounds = ((x >= 0) & (x < actual_width)
                 & (y >= 0) & (y < actual_height))

    padded_height, padded_width = depth_maps.shape[-2:]
    linear_indices = (
        y.clamp(0, padded_height - 1) * padded_width
        + x.clamp(0, padded_width - 1))
    query_count = reference_points.shape[1]
    patch_shape = (len(depth_maps), query_count, offset_x.numel())
    patch_depth = depth_maps.reshape(len(depth_maps), -1).gather(
        1, linear_indices.reshape(len(depth_maps), -1)).reshape(patch_shape)
    patch_valid = depth_valid.reshape(len(depth_maps), -1).gather(
        1, linear_indices.reshape(len(depth_maps), -1)).reshape(patch_shape)
    patch_valid &= active & in_bounds
    patch_valid &= torch.isfinite(patch_depth) & (patch_depth > 0)

    central = ((offset_x >= -1) & (offset_x <= 0)
               & (offset_y >= -1) & (offset_y <= 0))
    central_valid = patch_valid & central
    distance = offset_x.square() + offset_y.square()
    anchor_indices = distance.expand_as(patch_depth).masked_fill(
        ~central_valid, torch.iinfo(torch.long).max).argmin(dim=-1)
    anchors = patch_depth.gather(
        -1, anchor_indices[..., None]).squeeze(-1)
    has_anchor = central_valid.any(dim=-1)

    tolerance = torch.maximum(
        float(abs_depth_tolerance) / scales[:, None],
        float(rel_depth_tolerance) * anchors.abs())
    same_surface = patch_valid & (
        (patch_depth - anchors[..., None]).abs()
        <= tolerance[..., None])
    target_depth = patch_depth.masked_fill(
        ~same_surface, torch.nan).nanmedian(dim=-1).values
    target_valid = has_anchor & same_surface.any(dim=-1)
    target_valid &= torch.isfinite(target_depth) & (target_depth > 0)
    return target_depth, target_valid


def compute_confident_query_point_loss(
        reconstruction_outputs, cls_scores, bbox_preds,
        batch_data_samples, matches, score_thr=0.05, loss_weight=1.0,
        window_fraction=0.1, min_window_size=2, max_window_size=4,
        abs_depth_tolerance=0.05, rel_depth_tolerance=0.01):
    points = reconstruction_outputs['points_vggt']
    reference_points = reconstruction_outputs['reference_points_2d']
    pred_valid = reconstruction_outputs['valid_mask']
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError('reconstruction points must have shape [N, Q, 3]')
    if reference_points.shape != points.shape[:2] + (2,):
        raise ValueError('reference points must have shape [N, Q, 2]')
    if cls_scores.shape[:2] != points.shape[:2]:
        raise ValueError('classification scores must match reconstruction queries')
    if bbox_preds.shape != points.shape[:2] + (4,):
        raise ValueError('bbox predictions must have shape [N, Q, 4]')
    if len(batch_data_samples) != len(points) or len(matches) != len(points):
        raise ValueError('depth targets and matches must match flattened views')

    with torch.no_grad():
        positive_maps = [
            sample.token_positive_map for sample in batch_data_samples]
        class_scores = convert_grounding_to_cls_scores(
            cls_scores.detach().sigmoid(), positive_maps)
        foreground_scores = class_scores.amax(dim=-1)
        unmatched = torch.ones_like(foreground_scores, dtype=torch.bool)
        for view_index, (query_indices, _) in enumerate(matches):
            unmatched[view_index, query_indices] = False
        target_depth, target_valid = _sample_query_depth_targets(
            reference_points.detach(), bbox_preds.detach(),
            batch_data_samples, window_fraction,
            min_window_size, max_window_size,
            abs_depth_tolerance, rel_depth_tolerance)
        # GT-only target construction; predicted cameras never enter XYZ decoding.
        gt_extrinsics = torch.stack([
            sample.gt_extrinsics_vggt.to(points) for sample in batch_data_samples])
        gt_intrinsics = torch.stack([
            sample.gt_intrinsics.to(points) for sample in batch_data_samples])
        image_shapes = reference_points.new_tensor([
            sample.metainfo['img_shape'][:2] for sample in batch_data_samples])
        pixel_xy = reference_points.detach() * image_shapes[:, None, [1, 0]]
        _, target_points = unproject_query_depth(
            pixel_xy, torch.nan_to_num(target_depth), gt_extrinsics, gt_intrinsics)
        target_valid &= torch.isfinite(target_points).all(dim=-1)
        selected = foreground_scores > float(score_thr)
        selected &= unmatched & pred_valid.bool()
        selected &= target_valid
        selected &= (reference_points >= 0).all(dim=-1)
        selected &= (reference_points <= 1).all(dim=-1)

    prediction = points.float()
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('Non-finite reconstruction XYZ')
    errors = torch.linalg.vector_norm(
        prediction[selected] - target_points[selected], dim=-1)
    zero = prediction.sum() * 0.0
    return _distributed_valid_mean([errors], zero) * float(loss_weight)


def select_final_matching_outputs(all_layers_cls_scores,
                                  all_layers_bbox_preds,
                                  num_matching_queries):
    if num_matching_queries <= 0:
        raise ValueError('num_matching_queries must be positive')
    if all_layers_cls_scores.shape[:3] != all_layers_bbox_preds.shape[:3]:
        raise ValueError('classification and bbox outputs must share L, N, Q')
    if all_layers_cls_scores.shape[2] < num_matching_queries:
        raise ValueError('not enough decoder queries for reconstruction')
    return (
        all_layers_cls_scores[-1, :, -num_matching_queries:],
        all_layers_bbox_preds[-1, :, -num_matching_queries:],
    )


@torch.no_grad()
def match_reconstruction_queries(assigner, cls_scores, bbox_preds,
                                 batch_gt_instances, batch_img_metas):
    if cls_scores.shape[:2] != bbox_preds.shape[:2]:
        raise ValueError('classification and bbox predictions must share N, Q')
    if len(batch_gt_instances) != len(cls_scores):
        raise ValueError('GT instance count must match flattened views')
    if len(batch_img_metas) != len(cls_scores):
        raise ValueError('image metadata count must match flattened views')

    matches = []
    for cls_score, bbox_pred, gt_instances, img_meta in zip(
            cls_scores, bbox_preds, batch_gt_instances, batch_img_metas):
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
        pred_instances = InstanceData(
            scores=cls_score,
            bboxes=bbox_cxcywh_to_xyxy(bbox_pred) * factor)
        assign_result = assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        query_indices = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        gt_indices = assign_result.gt_inds[query_indices] - 1
        matches.append((query_indices, gt_indices.long()))
    return matches


def _distributed_valid_mean(errors, zero):
    valid_count = zero.new_tensor(float(sum(error.numel()
                                             for error in errors)))
    avg_factor = reduce_mean(valid_count)
    if errors:
        errors = torch.cat(errors)
        return errors.sum() / avg_factor.clamp_min(
            torch.finfo(avg_factor.dtype).tiny)
    return zero


def compute_matched_reconstruction_losses(
        reconstruction_outputs, batch_gt_instances, matches, point_weight=0.5):
    points = reconstruction_outputs['points_vggt']
    pred_valid = reconstruction_outputs['valid_mask']
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError('reconstruction points must have shape [N, Q, 3]')
    if pred_valid.shape != points.shape[:2]:
        raise ValueError('reconstruction valid mask must have shape [N, Q]')
    if len(batch_gt_instances) != len(points) or len(matches) != len(points):
        raise ValueError('reconstruction targets must match flattened views')
    if not torch.isfinite(points).all():
        raise FloatingPointError('Non-finite reconstruction XYZ')

    point_errors = []
    for view_index, (query_indices, gt_indices) in enumerate(matches):
        if query_indices.numel() == 0:
            continue
        gt_instances = batch_gt_instances[view_index]
        pred_points = points[view_index, query_indices].float()
        gt_points = gt_instances.centers_3d_vggt[gt_indices].float()
        point_valid = gt_instances.center_3d_valid_mask[gt_indices].bool().clone()
        point_valid &= pred_valid[view_index, query_indices].bool()
        point_valid &= torch.isfinite(gt_points).all(dim=-1)
        if point_valid.any():
            point_errors.append(torch.linalg.vector_norm(
                pred_points[point_valid] - gt_points[point_valid], dim=-1))

    point_zero = points.float().sum() * 0.0
    return {'loss_recon_point': (
        _distributed_valid_mean(point_errors, point_zero) * float(point_weight))}


class GroundingDINOQueryPointHead(nn.Module):
    """Decode camera-conditioned queries directly in normalized VGGT coordinates."""

    def __init__(self, query_dims=512, camera_dims=2048):
        super().__init__()
        self.camera_dims = int(camera_dims)
        # Same projection structure as VGGT CameraHead.camera_branch.
        self.camera_output_projection = nn.Sequential(
            nn.Linear(camera_dims, camera_dims // 2), nn.GELU(),
            nn.Linear(camera_dims // 2, query_dims))
        self.point_norm = nn.LayerNorm(query_dims)
        self.point_head = nn.Sequential(
            nn.Linear(query_dims, query_dims // 2), nn.GELU(),
            nn.Linear(query_dims // 2, 3))

    def forward(self, query, reference_points, camera_tokens):
        if query.ndim != 3 or reference_points.shape != query.shape[:2] + (2,):
            raise ValueError(
                'reference_points must have shape [N, Q, 2] matching query')
        if camera_tokens is None or camera_tokens.shape != (
                query.shape[0], self.camera_dims):
            raise ValueError('camera_tokens must have shape [N, camera_dims]')
        # Coordinate regression and the camera MLP run in FP32 under AMP.
        with torch.autocast(device_type=query.device.type, enabled=False):
            camera_features = self.camera_output_projection(camera_tokens.float())
            point_query = self.point_norm(query.float() + camera_features[:, None])
            points_vggt = self.point_head(point_query)
        if not torch.isfinite(points_vggt).all():
            raise FloatingPointError('Non-finite reconstruction XYZ')
        valid_mask = torch.isfinite(reference_points).all(dim=-1)
        valid_mask &= (reference_points >= 0).all(dim=-1)
        valid_mask &= (reference_points <= 1).all(dim=-1)
        return {
            'points_vggt': points_vggt,
            'reference_points_2d': reference_points,
            'valid_mask': valid_mask,
        }


@MODELS.register_module()
class ReconGroundingDINOHead(GroundingDINOHead):

    def __init__(self, contrastive_cfg=dict(max_text_len=256), **kwargs):
        instance_embedding_dims = int(kwargs.pop(
            'instance_embedding_dims', 128))
        if instance_embedding_dims <= 0:
            raise ValueError('instance_embedding_dims must be positive')
        self.reconstruction_dims = kwargs.pop('reconstruction_dims', 512)
        self.camera_dims = int(kwargs.pop('camera_dims', 2048))
        self.reconstruction_point_loss_weight = float(kwargs.pop(
            'reconstruction_point_loss_weight', 0.5))
        self.supervise_confident_query_point = bool(kwargs.pop(
            'supervise_confident_query_point', False))
        confident_point_cfg = kwargs.pop('confident_query_point_cfg', None)
        self.confident_query_point_cfg = dict(
            CONFIDENT_QUERY_POINT_DEFAULTS)
        if confident_point_cfg is not None:
            unknown_keys = (
                set(confident_point_cfg) - set(self.confident_query_point_cfg))
            if unknown_keys:
                raise ValueError(
                    'Unknown confident_query_point_cfg keys: '
                    f'{sorted(unknown_keys)}')
            self.confident_query_point_cfg.update(confident_point_cfg)
        super().__init__(contrastive_cfg=contrastive_cfg, **kwargs)
        self.instance_projection = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.GELU(),
            nn.Linear(self.embed_dims, instance_embedding_dims))
        self.reconstruction_head = GroundingDINOQueryPointHead(
            query_dims=self.reconstruction_dims, camera_dims=self.camera_dims)

    def forward(self, hidden_states: Tensor, references: List[Tensor],
                memory_text: Tensor, text_token_mask: Tensor):
        outputs = super().forward(
            hidden_states, references, memory_text, text_token_mask)
        self._last_cls_scores = outputs[0].detach()
        self._last_bbox_preds = outputs[1].detach()
        self._last_instance_embeddings = self.instance_projection(
            hidden_states[-1]).detach()
        return outputs

    def predict_reconstruction(self, reconstruction_hidden_states,
                               reference_points, camera_tokens,
                               instance_embeddings=None):
        outputs = self.reconstruction_head(
            reconstruction_hidden_states[-1], reference_points,
            camera_tokens)
        outputs['reconstruction_query'] = reconstruction_hidden_states[-1]
        outputs['hidden_states'] = reconstruction_hidden_states
        query_count = reference_points.shape[1]
        if instance_embeddings is None:
            instance_embeddings = self._last_instance_embeddings[
                :, -query_count:]
        if instance_embeddings.shape[:2] != (reference_points.shape[0],
                                              query_count):
            raise RuntimeError(
                'Instance embeddings must align with reconstruction queries')
        outputs['instance_embeddings_2d'] = instance_embeddings
        return outputs

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int],
             reconstruction_hidden_states: Tensor = None,
             reconstruction_outputs: Dict[str, Tensor] = None) -> dict:
        self.forward(hidden_states, references, memory_text, text_token_mask)
        losses = {}
        if reconstruction_outputs is None:
            return losses

        num_matching_queries = reconstruction_outputs['points_vggt'].shape[1]
        cls_scores, bbox_preds = select_final_matching_outputs(
            self._last_cls_scores, self._last_bbox_preds,
            num_matching_queries)
        batch_gt_instances = [
            data_sample.gt_instances for data_sample in batch_data_samples]
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples]
        matches = match_reconstruction_queries(
            self.assigner, cls_scores, bbox_preds,
            batch_gt_instances, batch_img_metas)
        reconstruction_outputs['reconstruction_matches'] = matches
        losses.update(compute_matched_reconstruction_losses(
            reconstruction_outputs, batch_gt_instances, matches,
            point_weight=self.reconstruction_point_loss_weight))
        if self.supervise_confident_query_point:
            losses['loss_confident_query_point'] = (
                compute_confident_query_point_loss(
                    reconstruction_outputs, cls_scores, bbox_preds,
                    batch_data_samples, matches,
                    **self.confident_query_point_cfg))
        return losses
