import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.dense_heads.grounding_dino_head import GroundingDINOHead
from mmdet.models.dense_heads.atss_vlfusion_head import (
    convert_grounding_to_cls_scores)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import reduce_mean
from mmengine.structures import InstanceData

from recondet.camera_alignment import unproject_query_depth


CONFIDENT_QUERY_DEPTH_DEFAULTS = dict(
    score_thr=0.05,
    loss_weight=1.0,
    window_fraction=0.1,
    min_window_size=2,
    max_window_size=4,
    abs_depth_tolerance=0.05,
    rel_depth_tolerance=0.01,
)

INSTANCE_CONSISTENCY_DEFAULTS = dict(
    embedding_dims=128,
    temperature=0.1,
    loss_weight=0.1,
    background_max_iou=0.3,
    background_ratio=2.0,
    min_background=8,
    max_background=32,
)


def sample_unmatched_background_queries(
        bbox_preds, batch_gt_instances, matches, batch_img_metas,
        max_iou=0.3, ratio=2.0, min_samples=8, max_samples=32):
    """Randomly sample unmatched queries that do not overlap a GT object."""
    if bbox_preds.ndim != 3 or bbox_preds.shape[-1] != 4:
        raise ValueError('bbox_preds must have shape [N, Q, 4]')
    num_views, num_queries = bbox_preds.shape[:2]
    if not (len(batch_gt_instances) == len(matches)
            == len(batch_img_metas) == num_views):
        raise ValueError('background sampling inputs must match N views')
    if not 0 <= max_iou <= 1:
        raise ValueError('max_iou must be between 0 and 1')
    if ratio < 0 or min_samples < 0 or max_samples < min_samples:
        raise ValueError('invalid background sampling limits')

    sampled_indices = []
    for view_index, ((matched_queries, _), gt_instances, img_meta) in enumerate(
            zip(matches, batch_gt_instances, batch_img_metas)):
        candidate_mask = torch.ones(
            num_queries, dtype=torch.bool, device=bbox_preds.device)
        candidate_mask[matched_queries] = False

        gt_bboxes = gt_instances.bboxes.to(bbox_preds.device)
        if len(gt_bboxes) > 0:
            img_h, img_w = img_meta['img_shape'][:2]
            factor = bbox_preds.new_tensor([img_w, img_h, img_w, img_h])
            pred_bboxes = (
                bbox_cxcywh_to_xyxy(bbox_preds[view_index]) * factor)
            max_overlaps = bbox_overlaps(
                pred_bboxes, gt_bboxes).amax(dim=1)
            candidate_mask &= max_overlaps < float(max_iou)

        candidates = torch.nonzero(
            candidate_mask, as_tuple=False).squeeze(-1)
        requested = max(
            int(min_samples),
            math.ceil(float(ratio) * matched_queries.numel()))
        sample_count = min(candidates.numel(), int(max_samples), requested)
        if sample_count:
            order = torch.randperm(
                candidates.numel(), device=candidates.device)[:sample_count]
            candidates = candidates[order]
        else:
            candidates = candidates[:0]
        sampled_indices.append(candidates)
    return sampled_indices


def compute_scene_local_supcon_loss(
        embeddings, instance_ids, scene_indices, view_indices, is_background,
        temperature=0.1):
    """Compute SupCon with cross-view positives and scene-local negatives."""
    if embeddings.ndim != 2:
        raise ValueError('embeddings must have shape [M, D]')
    expected_shape = (len(embeddings),)
    for name, value in (
            ('instance_ids', instance_ids),
            ('scene_indices', scene_indices),
            ('view_indices', view_indices),
            ('is_background', is_background)):
        if value.shape != expected_shape:
            raise ValueError(f'{name} must have shape [M]')
    if temperature <= 0:
        raise ValueError('temperature must be positive')

    zero = embeddings.float().sum() * 0.0
    if len(embeddings) == 0:
        return dict(
            loss=zero,
            anchor_count=zero.detach(),
            positive_pair_count=zero.detach(),
            positive_similarity=zero.detach(),
            negative_similarity=zero.detach())

    normalized = F.normalize(embeddings.float(), dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    valid_identity = (~is_background.bool()) & (instance_ids >= 0)
    same_scene = scene_indices[:, None] == scene_indices[None, :]
    same_instance = instance_ids[:, None] == instance_ids[None, :]
    different_view = view_indices[:, None] != view_indices[None, :]
    positive_mask = (
        same_scene & same_instance & different_view
        & valid_identity[:, None] & valid_identity[None, :])

    eye = torch.eye(
        len(embeddings), dtype=torch.bool, device=embeddings.device)
    same_instance_same_view = (
        same_scene & same_instance & ~different_view
        & valid_identity[:, None] & valid_identity[None, :])
    valid_candidate = valid_identity | is_background.bool()
    candidate_mask = (
        same_scene & ~eye & ~same_instance_same_view
        & valid_candidate[None, :])
    valid_anchor = valid_identity & positive_mask.any(dim=1)

    if not valid_anchor.any():
        return dict(
            loss=zero,
            anchor_count=zero.detach(),
            positive_pair_count=zero.detach(),
            positive_similarity=zero.detach(),
            negative_similarity=zero.detach())

    logits = similarity / float(temperature)
    log_denominator = logits.masked_fill(
        ~candidate_mask, -torch.inf).logsumexp(dim=1)
    positive_count = positive_mask.sum(dim=1).clamp_min(1)
    positive_logits = (logits * positive_mask).sum(dim=1) / positive_count
    loss = (log_denominator - positive_logits)[valid_anchor].mean()

    anchor_positive_mask = positive_mask & valid_anchor[:, None]
    negative_mask = (
        candidate_mask & ~positive_mask & valid_anchor[:, None])
    positive_similarity = similarity[anchor_positive_mask].mean()
    if negative_mask.any():
        negative_similarity = similarity[negative_mask].mean()
    else:
        negative_similarity = zero.detach()
    return dict(
        loss=loss,
        anchor_count=valid_anchor.sum().float().detach(),
        positive_pair_count=anchor_positive_mask.sum().float().detach(),
        positive_similarity=positive_similarity.detach(),
        negative_similarity=negative_similarity.detach())


def compute_matched_instance_consistency_loss(
        query_embeddings, batch_gt_instances, matches, background_indices,
        batch_data_samples, temperature=0.1):
    """Build scene-aware identity labels from final Hungarian assignments."""
    if query_embeddings.ndim != 3:
        raise ValueError('query_embeddings must have shape [N, Q, D]')
    num_views = len(query_embeddings)
    if not (len(batch_gt_instances) == len(matches)
            == len(background_indices) == len(batch_data_samples)
            == num_views):
        raise ValueError('instance consistency inputs must match N views')

    embedding_parts = []
    instance_id_parts = []
    scene_parts = []
    view_parts = []
    background_parts = []
    device = query_embeddings.device
    for view_embeddings, gt_instances, match, background, sample in zip(
            query_embeddings, batch_gt_instances, matches,
            background_indices, batch_data_samples):
        if not hasattr(sample, 'scene_batch_index') or not hasattr(
                sample, 'view_index'):
            raise RuntimeError(
                'instance consistency requires scene and view indices')
        scene_index = int(sample.scene_batch_index)
        view_index = int(sample.view_index)
        query_indices, gt_indices = match
        gt_instance_ids = gt_instances.instance_ids_3d.to(device)
        matched_ids = gt_instance_ids[gt_indices]
        valid = matched_ids >= 0
        if valid.any():
            identity_embeddings = view_embeddings[query_indices[valid]]
            identity_ids = matched_ids[valid].long()
            count = len(identity_embeddings)
            embedding_parts.append(identity_embeddings)
            instance_id_parts.append(identity_ids)
            scene_parts.append(torch.full(
                (count,), scene_index, dtype=torch.long, device=device))
            view_parts.append(torch.full(
                (count,), view_index, dtype=torch.long, device=device))
            background_parts.append(torch.zeros(
                count, dtype=torch.bool, device=device))
        if background.numel():
            background = background.to(device)
            count = background.numel()
            embedding_parts.append(view_embeddings[background])
            instance_id_parts.append(torch.full(
                (count,), -1, dtype=torch.long, device=device))
            scene_parts.append(torch.full(
                (count,), scene_index, dtype=torch.long, device=device))
            view_parts.append(torch.full(
                (count,), view_index, dtype=torch.long, device=device))
            background_parts.append(torch.ones(
                count, dtype=torch.bool, device=device))

    if embedding_parts:
        embeddings = torch.cat(embedding_parts)
        instance_ids = torch.cat(instance_id_parts)
        scene_indices = torch.cat(scene_parts)
        view_indices = torch.cat(view_parts)
        is_background = torch.cat(background_parts)
    else:
        embeddings = query_embeddings.reshape(-1, query_embeddings.shape[-1])[:0]
        instance_ids = torch.empty(0, dtype=torch.long, device=device)
        scene_indices = torch.empty(0, dtype=torch.long, device=device)
        view_indices = torch.empty(0, dtype=torch.long, device=device)
        is_background = torch.empty(0, dtype=torch.bool, device=device)

    result = compute_scene_local_supcon_loss(
        embeddings, instance_ids, scene_indices, view_indices,
        is_background, temperature=temperature)
    result['background_count'] = is_background.sum().float().detach()
    return result


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
                f'Confident query depth supervision requires {missing}')
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


def compute_confident_query_depth_loss(
        reconstruction_outputs, cls_scores, bbox_preds,
        batch_data_samples, matches, score_thr=0.05, loss_weight=1.0,
        window_fraction=0.1, min_window_size=2, max_window_size=4,
        abs_depth_tolerance=0.05, rel_depth_tolerance=0.01):
    depth = reconstruction_outputs['depth']
    reference_points = reconstruction_outputs['reference_points_2d']
    pred_valid = reconstruction_outputs['valid_mask']
    if depth.ndim != 3 or depth.shape[-1] != 1:
        raise ValueError('reconstruction depth must have shape [N, Q, 1]')
    if reference_points.shape != depth.shape[:2] + (2,):
        raise ValueError('reference points must have shape [N, Q, 2]')
    if cls_scores.shape[:2] != depth.shape[:2]:
        raise ValueError('classification scores must match reconstruction queries')
    if bbox_preds.shape != depth.shape[:2] + (4,):
        raise ValueError('bbox predictions must have shape [N, Q, 4]')
    if len(batch_data_samples) != len(depth) or len(matches) != len(depth):
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
        selected = foreground_scores > float(score_thr)
        selected &= unmatched & pred_valid.bool()
        selected &= target_valid
        selected &= (reference_points >= 0).all(dim=-1)
        selected &= (reference_points <= 1).all(dim=-1)

    prediction = depth[..., 0].float()
    selected &= torch.isfinite(prediction) & (prediction > 0)
    errors = (prediction - target_depth).abs()[selected]
    zero = torch.nan_to_num(prediction).sum() * 0.0
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
        reconstruction_outputs, batch_gt_instances, matches,
        depth_weight=1.0, point_weight=0.5):
    depth = reconstruction_outputs['depth']
    points = reconstruction_outputs['points_vggt']
    pred_valid = reconstruction_outputs['valid_mask']
    if depth.ndim != 3 or depth.shape[-1] != 1:
        raise ValueError('reconstruction depth must have shape [N, Q, 1]')
    if points.shape != depth.shape[:2] + (3,):
        raise ValueError('reconstruction points must have shape [N, Q, 3]')
    if pred_valid.shape != depth.shape[:2]:
        raise ValueError('reconstruction valid mask must have shape [N, Q]')
    if len(batch_gt_instances) != len(depth) or len(matches) != len(depth):
        raise ValueError('reconstruction targets must match flattened views')

    depth_errors = []
    point_errors = []
    for view_index, (query_indices, gt_indices) in enumerate(matches):
        if query_indices.numel() == 0:
            continue
        gt_instances = batch_gt_instances[view_index]
        pred_depth = depth[view_index, query_indices, 0].float()
        gt_depth = gt_instances.center_depth_vggt[gt_indices].float()
        depth_valid = gt_instances.center_depth_valid_mask[gt_indices].bool()
        depth_valid &= torch.isfinite(pred_depth)
        depth_valid &= torch.isfinite(gt_depth) & (gt_depth > 0)
        if depth_valid.any():
            depth_errors.append(
                (pred_depth[depth_valid] - gt_depth[depth_valid]).abs())

        pred_points = points[view_index, query_indices].float()
        gt_points = gt_instances.centers_3d_vggt[gt_indices].float()
        point_valid = gt_instances.center_3d_valid_mask[gt_indices].bool()
        point_valid &= pred_valid[view_index, query_indices].bool()
        point_valid &= torch.isfinite(pred_points).all(dim=-1)
        point_valid &= torch.isfinite(gt_points).all(dim=-1)
        if point_valid.any():
            point_errors.append(torch.linalg.vector_norm(
                pred_points[point_valid] - gt_points[point_valid], dim=-1))

    depth_zero = torch.nan_to_num(depth.float()).sum() * 0.0
    point_zero = torch.nan_to_num(points.float()).sum() * 0.0
    return {
        'loss_recon_depth': (
            _distributed_valid_mean(depth_errors, depth_zero)
            * float(depth_weight)),
        'loss_recon_point': (
            _distributed_valid_mean(point_errors, point_zero)
            * float(point_weight)),
    }


class GroundingDINOQueryDepthHead(nn.Module):

    def __init__(self, query_dims=512, log_depth_range=(-6.0, 6.0)):
        super().__init__()
        if (len(log_depth_range) != 2 or
                log_depth_range[0] >= log_depth_range[1]):
            raise ValueError('log_depth_range must be an increasing pair')
        self.log_depth_range = tuple(float(value)
                                     for value in log_depth_range)
        self.depth_head = nn.Sequential(
            nn.LayerNorm(query_dims),
            nn.Linear(query_dims, query_dims // 2),
            nn.GELU(),
            nn.Linear(query_dims // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1))
        with torch.no_grad():
            nn.init.zeros_(self.depth_head[-1].weight)
            nn.init.zeros_(self.depth_head[-1].bias)

    def forward(self, query, reference_points, extrinsics, intrinsics,
                image_shapes):
        if reference_points.shape != query.shape[:2] + (2,):
            raise ValueError(
                'reference_points must have shape [N, Q, 2] matching query')
        if image_shapes.shape != (query.shape[0], 2):
            raise ValueError('image_shapes must have shape [N, 2]')

        depth_logits = self.depth_head(query)
        with torch.autocast(device_type=query.device.type, enabled=False):
            depth_logits_fp32 = depth_logits.float()
            bounded_logits = depth_logits_fp32.clamp(*self.log_depth_range)
            bounded_logits = depth_logits_fp32 + (
                bounded_logits - depth_logits_fp32).detach()
            depth = bounded_logits.exp()

            image_shapes = image_shapes.to(
                device=query.device, dtype=torch.float32)
            pixel_scale = torch.stack(
                [image_shapes[:, 1], image_shapes[:, 0]], dim=-1)
            pixel_xy = reference_points.float() * pixel_scale[:, None]
            points_camera, points_vggt = unproject_query_depth(
                pixel_xy, depth, extrinsics, intrinsics)

        valid_mask = torch.isfinite(points_vggt).all(dim=-1)
        valid_mask &= torch.isfinite(depth[..., 0]) & (depth[..., 0] > 0)
        valid_mask &= (reference_points >= 0).all(dim=-1)
        valid_mask &= (reference_points <= 1).all(dim=-1)
        return {
            'depth_logits': depth_logits,
            'depth': depth,
            'points_camera': points_camera,
            'points_vggt': points_vggt,
            'reference_points_2d': reference_points,
            'valid_mask': valid_mask,
        }


@MODELS.register_module()
class ReconGroundingDINOHead(GroundingDINOHead):

    def __init__(self, contrastive_cfg=dict(max_text_len=256), **kwargs):
        self.supervise_2d_bbox = bool(kwargs.pop(
            'supervise_2d_bbox', True))
        self.reconstruction_dims = kwargs.pop('reconstruction_dims', 512)
        self.log_depth_range = kwargs.pop(
            'log_depth_range', (-6.0, 6.0))
        self.reconstruction_depth_loss_weight = float(kwargs.pop(
            'reconstruction_depth_loss_weight', 1.0))
        self.reconstruction_point_loss_weight = float(kwargs.pop(
            'reconstruction_point_loss_weight', 0.5))
        self.supervise_confident_query_depth = bool(kwargs.pop(
            'supervise_confident_query_depth', False))
        confident_depth_cfg = kwargs.pop('confident_query_depth_cfg', None)
        self.supervise_instance_consistency = bool(kwargs.pop(
            'supervise_instance_consistency', False))
        instance_consistency_cfg = kwargs.pop(
            'instance_consistency_cfg', None)
        self.instance_consistency_cfg = dict(INSTANCE_CONSISTENCY_DEFAULTS)
        if instance_consistency_cfg is not None:
            unknown_keys = (
                set(instance_consistency_cfg)
                - set(self.instance_consistency_cfg))
            if unknown_keys:
                raise ValueError(
                    'Unknown instance_consistency_cfg keys: '
                    f'{sorted(unknown_keys)}')
            self.instance_consistency_cfg.update(instance_consistency_cfg)
        self.confident_query_depth_cfg = dict(
            CONFIDENT_QUERY_DEPTH_DEFAULTS)
        if confident_depth_cfg is not None:
            unknown_keys = (
                set(confident_depth_cfg) - set(self.confident_query_depth_cfg))
            if unknown_keys:
                raise ValueError(
                    'Unknown confident_query_depth_cfg keys: '
                    f'{sorted(unknown_keys)}')
            self.confident_query_depth_cfg.update(confident_depth_cfg)
        super().__init__(contrastive_cfg=contrastive_cfg, **kwargs)
        embedding_dims = int(
            self.instance_consistency_cfg['embedding_dims'])
        if embedding_dims <= 0:
            raise ValueError('embedding_dims must be positive')
        self.instance_projection = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.GELU(),
            nn.Linear(self.embed_dims, embedding_dims))
        self.reconstruction_head = GroundingDINOQueryDepthHead(
            query_dims=self.reconstruction_dims,
            log_depth_range=self.log_depth_range)

    def forward(self, hidden_states: Tensor, references: List[Tensor],
                memory_text: Tensor, text_token_mask: Tensor):
        outputs = super().forward(
            hidden_states, references, memory_text, text_token_mask)
        self._last_cls_scores = outputs[0].detach()
        self._last_bbox_preds = outputs[1].detach()
        return outputs

    def predict_reconstruction(self, reconstruction_hidden_states,
                               reference_points, extrinsics, intrinsics,
                               image_shapes):
        outputs = self.reconstruction_head(
            reconstruction_hidden_states[-1], reference_points,
            extrinsics, intrinsics, image_shapes)
        outputs['reconstruction_query'] = reconstruction_hidden_states[-1]
        outputs['hidden_states'] = reconstruction_hidden_states
        return outputs

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int],
             reconstruction_hidden_states: Tensor = None,
             reconstruction_outputs: Dict[str, Tensor] = None) -> dict:
        detection_loss_enabled = getattr(
            self, 'supervise_2d_bbox', True)
        if detection_loss_enabled:
            losses = super().loss(
                hidden_states=hidden_states,
                references=references,
                memory_text=memory_text,
                text_token_mask=text_token_mask,
                enc_outputs_class=enc_outputs_class,
                enc_outputs_coord=enc_outputs_coord,
                batch_data_samples=batch_data_samples,
                dn_meta=dn_meta)
        else:
            self.forward(
                hidden_states, references, memory_text, text_token_mask)
            losses = {}
        if self.supervise_instance_consistency:
            num_denoising_queries = (
                int(dn_meta['num_denoising_queries'])
                if dn_meta is not None else 0)
            matching_hidden_states = hidden_states[
                -1, :, num_denoising_queries:]
            matching_cls_scores = self._last_cls_scores[
                -1, :, num_denoising_queries:]
            matching_bbox_preds = self._last_bbox_preds[
                -1, :, num_denoising_queries:]
            batch_gt_instances = [
                sample.gt_instances for sample in batch_data_samples]
            batch_img_metas = [
                sample.metainfo for sample in batch_data_samples]
            matches = match_reconstruction_queries(
                self.assigner, matching_cls_scores, matching_bbox_preds,
                batch_gt_instances, batch_img_metas)
            cfg = self.instance_consistency_cfg
            background_indices = sample_unmatched_background_queries(
                matching_bbox_preds, batch_gt_instances, matches,
                batch_img_metas,
                max_iou=cfg['background_max_iou'],
                ratio=cfg['background_ratio'],
                min_samples=cfg['min_background'],
                max_samples=cfg['max_background'])
            query_embeddings = self.instance_projection(
                matching_hidden_states)
            instance_result = compute_matched_instance_consistency_loss(
                query_embeddings, batch_gt_instances, matches,
                background_indices, batch_data_samples,
                temperature=cfg['temperature'])
            losses['loss_instance_consistency'] = (
                instance_result['loss'] * float(cfg['loss_weight']))
            for name in (
                    'anchor_count', 'positive_pair_count', 'background_count',
                    'positive_similarity', 'negative_similarity'):
                losses[f'instance_{name}'] = instance_result[name]
        if reconstruction_outputs is None:
            return losses

        num_matching_queries = reconstruction_outputs['depth'].shape[1]
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
        losses.update(compute_matched_reconstruction_losses(
            reconstruction_outputs, batch_gt_instances, matches,
            depth_weight=self.reconstruction_depth_loss_weight,
            point_weight=self.reconstruction_point_loss_weight))
        if self.supervise_confident_query_depth:
            losses['loss_confident_query_depth'] = (
                compute_confident_query_depth_loss(
                    reconstruction_outputs, cls_scores, bbox_preds,
                    batch_data_samples, matches,
                    **self.confident_query_depth_cfg))
        return losses
