import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.utils import reduce_mean
from mmdet3d.registry import MODELS
from torch import Tensor

from recondet.camera_alignment import aligned_boxes_to_vggt
from mmdet3d.structures.ops.iou3d_calculator import (
    axis_aligned_bbox_overlaps_3d)


def supervised_instance_contrastive_loss(
        embeddings: Tensor, group_keys: Tensor, view_indices: Tensor,
        temperature: float = 0.07) -> Tensor:
    """Supervised contrastive loss with positives from different views."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if embeddings.ndim != 2:
        raise ValueError('embeddings must have shape [M, C]')
    if group_keys.shape != (len(embeddings), 2):
        raise ValueError('group_keys must have shape [M, 2]')
    if view_indices.shape != (len(embeddings),):
        raise ValueError('view_indices must have shape [M]')
    if len(embeddings) < 2:
        return embeddings.sum() * 0.0

    normalized = F.normalize(embeddings.float(), dim=-1)
    logits = normalized @ normalized.t() / float(temperature)
    identity = torch.eye(
        len(embeddings), dtype=torch.bool, device=embeddings.device)
    same_instance = (group_keys[:, None] == group_keys[None, :]).all(dim=-1)
    positives = same_instance & (
        view_indices[:, None] != view_indices[None, :])
    candidates = ~identity
    valid_anchors = positives.any(dim=-1)
    if not valid_anchors.any():
        return embeddings.sum() * 0.0

    logits = logits.masked_fill(~candidates, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    positive_count = positives.sum(dim=-1).clamp_min(1)
    per_anchor = -(log_prob.masked_fill(~positives, 0.0).sum(dim=-1)
                   / positive_count)
    return per_anchor[valid_anchors].mean()


def _center_size_to_minmax(centers: Tensor, sizes: Tensor) -> Tensor:
    half_sizes = sizes / 2.0
    return torch.cat([centers - half_sizes, centers + half_sizes], dim=-1)


class ReconstructionObjectHead(nn.Module):
    """Predict one VGGT-frame 3D box for every reconstruction query."""

    def __init__(
            self, query_dims: int, semantic_dims: int, num_classes: int,
            hidden_dims: int = 256, instance_dims: int = 128,
            temperature: float = 0.07, instance_weight: float = 0.2,
            cls_weight: float = 1.0, center_weight: float = 0.5,
            size_weight: float = 1.0, giou_weight: float = 0.5,
            size_epsilon: float = 1e-5,
            cls_loss: dict = dict(
                type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0,
                alpha=0.25, loss_weight=1.0)):
        super().__init__()
        if min(query_dims, semantic_dims, hidden_dims,
               instance_dims, num_classes) <= 0:
            raise ValueError('head dimensions and class count must be positive')
        if size_epsilon <= 0:
            raise ValueError('size_epsilon must be positive')
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.instance_weight = float(instance_weight)
        self.cls_weight = float(cls_weight)
        self.center_weight = float(center_weight)
        self.size_weight = float(size_weight)
        self.giou_weight = float(giou_weight)
        self.size_epsilon = float(size_epsilon)
        self.cls_loss = MODELS.build(cls_loss)

        def prediction_head(output_dims):
            return nn.Sequential(
                nn.LayerNorm(query_dims),
                nn.Linear(query_dims, hidden_dims), nn.GELU(),
                nn.Linear(hidden_dims, output_dims))

        self.instance_projection = prediction_head(instance_dims)
        self.center_offset_head = prediction_head(3)
        self.size_head = prediction_head(3)
        self.semantic_projection = nn.Linear(semantic_dims, query_dims)
        self.class_fusion_norm = nn.LayerNorm(query_dims)
        self.class_head = prediction_head(num_classes)
        with torch.no_grad():
            nn.init.zeros_(self.center_offset_head[-1].weight)
            nn.init.zeros_(self.center_offset_head[-1].bias)
            nn.init.zeros_(self.class_head[-1].weight)
            nn.init.constant_(
                self.class_head[-1].bias,
                -math.log((1.0 - 0.01) / 0.01))

    def forward(self, queries_3d: Tensor, queries_2d: Tensor,
                points_vggt: Tensor) -> Dict[str, Tensor]:
        if queries_3d.ndim != 3:
            raise ValueError('3D queries must have shape [N, Q, C]')
        if queries_2d.ndim != 3 or queries_2d.shape[:2] != queries_3d.shape[:2]:
            raise ValueError('2D queries must share the [N, Q] dimensions')
        if points_vggt.shape != queries_3d.shape[:2] + (3,):
            raise ValueError('VGGT points must have shape [N, Q, 3]')

        queries_3d_fp32 = queries_3d.float()
        fused = self.class_fusion_norm(
            queries_3d_fp32 + self.semantic_projection(queries_2d.float()))
        center_offsets = self.center_offset_head(queries_3d_fp32).float()
        sizes = F.softplus(self.size_head(queries_3d_fp32).float())
        sizes = sizes + self.size_epsilon
        logits = self.class_head(fused).float()
        scores, labels = logits.sigmoid().max(dim=-1)
        return {
            'fused_detection_query': fused,
            'bbox_center_offsets_vggt': center_offsets,
            'bbox_centers_vggt': points_vggt.float() + center_offsets,
            'bbox_sizes_vggt': sizes,
            'bbox_logits': logits,
            'bbox_scores': scores,
            'bbox_labels': labels,
            'instance_embeddings': self.instance_projection(
                queries_3d_fp32),
        }

    @staticmethod
    def _batch_item(value, index):
        if isinstance(value, Tensor):
            return value[index]
        return value[index]

    def _build_targets(self, predictions, matches, batch_data_samples,
                       num_views, first_frame_pose, axis_align_matrix,
                       scene_scale, logger):
        logits = predictions['bbox_logits']
        num_flat_views, num_queries = logits.shape[:2]
        if num_views <= 0 or num_flat_views != len(batch_data_samples) * num_views:
            raise ValueError('predictions must contain B * num_views entries')
        if len(matches) != num_flat_views:
            raise ValueError('2D matches must align with flattened views')

        device = logits.device
        cls_targets = torch.full(
            (num_flat_views, num_queries), self.num_classes,
            dtype=torch.long, device=device)
        bbox_centers = logits.new_zeros(num_flat_views, num_queries, 3)
        bbox_sizes = logits.new_zeros(num_flat_views, num_queries, 3)
        bbox_valid = torch.zeros(
            num_flat_views, num_queries, dtype=torch.bool, device=device)
        identity_flat_indices = []
        identity_keys = []
        identity_views = []
        matched_query_indices = []
        errors = []

        scene_targets = []
        for scene_index, sample in enumerate(batch_data_samples):
            gt_3d = sample.gt_instances_3d
            missing_id_field = not hasattr(gt_3d, 'instance_ids_3d')
            ids_3d = torch.as_tensor(
                getattr(gt_3d, 'instance_ids_3d', []),
                device=device, dtype=torch.long)
            labels_3d = gt_3d.labels_3d.to(device=device, dtype=torch.long)
            centers_aligned = gt_3d.bboxes_3d.gravity_center.to(device)
            sizes_aligned = gt_3d.bboxes_3d.tensor[:, 3:6].to(device)
            centers_vggt, sizes_vggt = aligned_boxes_to_vggt(
                centers_aligned, sizes_aligned,
                self._batch_item(first_frame_pose, scene_index),
                self._batch_item(axis_align_matrix, scene_index),
                self._batch_item(scene_scale, scene_index))
            id_counts = {int(value): int((ids_3d == value).sum())
                         for value in ids_3d.unique()}
            id_to_index = {int(value): index
                           for index, value in enumerate(ids_3d.tolist())
                           if id_counts[int(value)] == 1}
            duplicate_ids = {key for key, count in id_counts.items()
                             if count != 1}
            scene_targets.append((id_to_index, duplicate_ids,
                                  missing_id_field, labels_3d,
                                  centers_vggt, sizes_vggt))

        for flat_view, (query_indices, gt_indices) in enumerate(matches):
            scene_index = flat_view // num_views
            view_index = flat_view % num_views
            sample = batch_data_samples[scene_index]
            gt_2d = sample.gt_instances_2d[view_index]
            query_indices = query_indices.to(device=device, dtype=torch.long)
            gt_indices = gt_indices.to(device=device, dtype=torch.long)
            labels_2d = gt_2d.labels.to(device=device, dtype=torch.long)[
                gt_indices]
            cls_targets[flat_view, query_indices] = labels_2d
            matched_query_indices.append(query_indices)

            if not hasattr(gt_2d, 'instance_ids_3d'):
                errors.append(
                    f'scene={sample.metainfo.get("scene_id", scene_index)} '
                    f'view={view_index}: 2D GT has no instance_ids_3d')
                continue
            ids_2d = gt_2d.instance_ids_3d.to(
                device=device, dtype=torch.long)[gt_indices]
            flat_indices = flat_view * num_queries + query_indices
            valid_identity = ids_2d >= 0
            identity_flat_indices.append(flat_indices[valid_identity])
            identity_keys.append(torch.stack([
                torch.full_like(ids_2d[valid_identity], scene_index),
                ids_2d[valid_identity]], dim=-1))
            identity_views.append(torch.full_like(
                ids_2d[valid_identity], view_index))

            (id_to_index, duplicate_ids, missing_id_field, labels_3d,
             centers_vggt, sizes_vggt) = scene_targets[scene_index]
            for query_index, gt_index, label_2d, instance_id in zip(
                    query_indices.tolist(), gt_indices.tolist(),
                    labels_2d.tolist(), ids_2d.tolist()):
                context = (
                    f'scene={sample.metainfo.get("scene_id", scene_index)} '
                    f'view={view_index} query={query_index} gt_2d={gt_index} '
                    f'instance_id={instance_id}')
                if missing_id_field:
                    errors.append(
                        f'{context}: 3D GT has no instance_ids_3d')
                    continue
                if instance_id in duplicate_ids:
                    errors.append(f'{context}: duplicate 3D instance id')
                    continue
                box_index = id_to_index.get(instance_id)
                if box_index is None:
                    errors.append(f'{context}: missing 3D instance id')
                    continue
                label_3d = int(labels_3d[box_index])
                if label_3d != label_2d:
                    errors.append(
                        f'{context}: label mismatch 2D={label_2d} 3D={label_3d}')
                    continue
                bbox_centers[flat_view, query_index] = centers_vggt[box_index]
                bbox_sizes[flat_view, query_index] = sizes_vggt[box_index]
                bbox_valid[flat_view, query_index] = True

        if errors:
            logger.error(
                'Reconstruction bbox target mapping errors; affected box '
                'targets were skipped:\n' + '\n'.join(errors))
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        return {
            'cls_targets': cls_targets,
            'bbox_centers_vggt': bbox_centers,
            'bbox_sizes_vggt': bbox_sizes,
            'bbox_valid': bbox_valid,
            'identity_flat_indices': (
                torch.cat(identity_flat_indices)
                if identity_flat_indices else empty_long),
            'identity_keys': (
                torch.cat(identity_keys) if identity_keys
                else torch.empty(0, 2, dtype=torch.long, device=device)),
            'identity_views': (
                torch.cat(identity_views) if identity_views else empty_long),
            'matched_query_indices': (
                torch.cat(matched_query_indices)
                if matched_query_indices else empty_long),
        }

    def loss(self, predictions, matches, batch_data_samples, num_views,
             first_frame_pose, axis_align_matrix, scene_scale, logger=None):
        if logger is None:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
        targets = self._build_targets(
            predictions, matches, batch_data_samples, num_views,
            first_frame_pose, axis_align_matrix, scene_scale, logger)
        logits = predictions['bbox_logits']
        zero = logits.sum() * 0.0
        local_num_matches = int(targets['matched_query_indices'].numel())
        cls_avg_factor = reduce_mean(logits.new_tensor(
            [local_num_matches], dtype=torch.float32)).clamp_min(1.0).item()
        cls_loss = self.cls_loss(
            logits.reshape(-1, self.num_classes),
            targets['cls_targets'].reshape(-1),
            avg_factor=cls_avg_factor) * self.cls_weight

        flat_embeddings = predictions['instance_embeddings'].reshape(
            -1, predictions['instance_embeddings'].shape[-1])
        identity_indices = targets['identity_flat_indices']
        instance_loss = supervised_instance_contrastive_loss(
            flat_embeddings[identity_indices], targets['identity_keys'],
            targets['identity_views'], self.temperature)

        valid = targets['bbox_valid']
        local_num_boxes = int(valid.sum())
        bbox_avg_factor = reduce_mean(logits.new_tensor(
            [local_num_boxes], dtype=torch.float32)).clamp_min(1.0).item()
        if valid.any():
            pred_centers = predictions['bbox_centers_vggt'][valid]
            pred_sizes = predictions['bbox_sizes_vggt'][valid]
            target_centers = targets['bbox_centers_vggt'][valid]
            target_sizes = targets['bbox_sizes_vggt'][valid]
            center_loss = F.smooth_l1_loss(
                pred_centers, target_centers, reduction='sum') / bbox_avg_factor
            size_loss = F.smooth_l1_loss(
                pred_sizes, target_sizes, reduction='sum') / bbox_avg_factor
            pred_boxes = _center_size_to_minmax(pred_centers, pred_sizes)
            target_boxes = _center_size_to_minmax(
                target_centers, target_sizes)
            giou = axis_aligned_bbox_overlaps_3d(
                pred_boxes.unsqueeze(0), target_boxes.unsqueeze(0),
                mode='giou', is_aligned=True).squeeze(0)
            giou_loss = (1.0 - giou).sum() / bbox_avg_factor
        else:
            center_loss = zero
            size_loss = zero
            giou_loss = zero

        losses = {
            'loss_recon_instance': instance_loss * self.instance_weight,
            'loss_recon_bbox_cls': cls_loss,
            'loss_recon_bbox_center': center_loss * self.center_weight,
            'loss_recon_bbox_size': size_loss * self.size_weight,
            'loss_recon_bbox_giou': giou_loss * self.giou_weight,
        }
        diagnostics = {
            'num_2d_matches': local_num_matches,
            'num_bbox_targets': local_num_boxes,
            'matched_query_indices': targets['matched_query_indices'].detach(),
        }
        return losses, diagnostics
