import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.utils import reduce_mean
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops.iou3d_calculator import (
    axis_aligned_bbox_overlaps_3d)
from torch import Tensor


def _center_size_to_minmax(centers: Tensor, sizes: Tensor) -> Tensor:
    half_sizes = sizes / 2.0
    return torch.cat([centers - half_sizes, centers + half_sizes], dim=-1)


class ReconstructionObjectHead(nn.Module):
    """Predict one aligned 3D box for every scene-level query cluster."""

    def __init__(self, query_dims: int, semantic_dims: int, num_classes: int,
                 hidden_dims: int = 256, size_logit_range=(-5.0, 5.0),
                 cls_loss: dict = dict(
                     type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0,
                     alpha=0.25, loss_weight=1.0)):
        super().__init__()
        if min(query_dims, semantic_dims, hidden_dims, num_classes) <= 0:
            raise ValueError('head dimensions and class count must be positive')
        if (len(size_logit_range) != 2
                or size_logit_range[0] >= size_logit_range[1]):
            raise ValueError('size_logit_range must be an increasing pair')
        self.num_classes = int(num_classes)
        self.size_logit_range = tuple(float(v) for v in size_logit_range)
        self.cls_loss = MODELS.build(cls_loss)

        def prediction_head(output_dims):
            return nn.Sequential(
                nn.LayerNorm(query_dims),
                nn.Linear(query_dims, hidden_dims), nn.GELU(),
                nn.Linear(hidden_dims, output_dims))

        self.semantic_projection = nn.Linear(semantic_dims, query_dims)
        self.query_fusion_norm = nn.LayerNorm(query_dims)
        self.center_offset_head = prediction_head(3)
        self.size_log_head = prediction_head(3)
        self.class_head = prediction_head(num_classes)
        with torch.no_grad():
            nn.init.zeros_(self.center_offset_head[-1].weight)
            nn.init.zeros_(self.center_offset_head[-1].bias)
            nn.init.zeros_(self.size_log_head[-1].weight)
            nn.init.zeros_(self.size_log_head[-1].bias)
            nn.init.zeros_(self.class_head[-1].weight)
            nn.init.constant_(
                self.class_head[-1].bias,
                -math.log((1.0 - 0.01) / 0.01))

    def forward(self, queries_3d: Tensor, queries_2d: Tensor,
                cluster_centers: Tensor) -> Dict[str, Tensor]:
        if queries_3d.ndim != 3:
            raise ValueError('3D queries must have shape [B, K, C]')
        if queries_2d.ndim != 3 or queries_2d.shape[:2] != queries_3d.shape[:2]:
            raise ValueError('2D queries must share the [B, K] dimensions')
        if cluster_centers.shape != queries_3d.shape[:2] + (3,):
            raise ValueError('cluster centers must have shape [B, K, 3]')

        queries_3d = queries_3d.float()
        fused = self.query_fusion_norm(
            queries_3d + self.semantic_projection(queries_2d.float()))
        center_offsets = self.center_offset_head(fused).float()
        raw_size_logs = self.size_log_head(fused).float()
        bounded_size_logs = raw_size_logs.clamp(*self.size_logit_range)
        size_logs = raw_size_logs + (
            bounded_size_logs - raw_size_logs).detach()
        sizes = size_logs.exp()
        logits = self.class_head(fused).float()
        scores, labels = logits.sigmoid().max(dim=-1)
        return {
            'fused_detection_query': fused,
            'bbox_center_offsets_aligned': center_offsets,
            'bbox_centers_aligned': cluster_centers.float() + center_offsets,
            'bbox_size_logs': size_logs,
            'bbox_sizes_aligned': sizes,
            'bbox_logits': logits,
            'bbox_scores': scores,
            'bbox_labels': labels,
        }

    def loss(self, predictions, batch_data_samples, matcher, loss_weights):
        centers = predictions['bbox_centers_aligned']
        sizes = predictions['bbox_sizes_aligned']
        size_logs = predictions['bbox_size_logs']
        logits = predictions['bbox_logits']
        if centers.ndim != 3 or centers.shape[-1] != 3:
            raise ValueError('predicted centers must have shape [B, K, 3]')
        if sizes.shape != centers.shape or size_logs.shape != centers.shape:
            raise ValueError('predicted sizes must match predicted centers')
        if logits.shape[:2] != centers.shape[:2]:
            raise ValueError('classification logits must match box predictions')
        if len(batch_data_samples) != centers.shape[0]:
            raise ValueError('one data sample is required per scene')

        matches = []
        for batch_id, sample in enumerate(batch_data_samples):
            gt = sample.gt_instances_3d
            gt_sizes = gt.bboxes_3d.tensor[:, 3:6].clamp_min(1e-5)
            matches.append(matcher._get_targets(
                centers[batch_id], sizes[batch_id], size_logs[batch_id],
                logits[batch_id], gt.bboxes_3d.gravity_center,
                gt_sizes, gt.labels_3d))
        local_num_pos = sum(pred.numel() for pred, _ in matches)
        avg_factor = reduce_mean(centers.new_tensor(
            [local_num_pos], dtype=torch.float32)).clamp_min(1.0).item()

        center_losses = []
        size_losses = []
        cls_losses = []
        giou_losses = []
        center_min = matcher.center_min.to(centers)
        center_extent = matcher.center_extent.to(centers)
        for batch_id, (pred_indices, gt_indices) in enumerate(matches):
            sample = batch_data_samples[batch_id]
            gt = sample.gt_instances_3d
            gt_centers = gt.bboxes_3d.gravity_center
            gt_sizes = gt.bboxes_3d.tensor[:, 3:6].clamp_min(1e-5)
            cls_target = torch.full(
                (centers.shape[1],), self.num_classes,
                dtype=torch.long, device=centers.device)
            cls_target[pred_indices] = gt.labels_3d[gt_indices]
            cls_losses.append(self.cls_loss(
                logits[batch_id], cls_target, avg_factor=avg_factor))
            if pred_indices.numel() == 0:
                center_losses.append(centers[batch_id].sum() * 0.0)
                size_losses.append(size_logs[batch_id].sum() * 0.0)
                giou_losses.append(sizes[batch_id].sum() * 0.0)
                continue
            pred_centers = centers[batch_id, pred_indices]
            matched_gt_centers = gt_centers[gt_indices]
            center_losses.append(F.l1_loss(
                (pred_centers - center_min) / center_extent,
                (matched_gt_centers - center_min) / center_extent,
                reduction='sum') / avg_factor)
            size_losses.append(F.l1_loss(
                size_logs[batch_id, pred_indices],
                gt_sizes[gt_indices].log(), reduction='sum') / avg_factor)
            pred_boxes = _center_size_to_minmax(
                pred_centers, sizes[batch_id, pred_indices])
            gt_boxes = _center_size_to_minmax(
                matched_gt_centers, gt_sizes[gt_indices])
            giou = axis_aligned_bbox_overlaps_3d(
                pred_boxes.unsqueeze(0), gt_boxes.unsqueeze(0),
                mode='giou', is_aligned=True)
            giou_losses.append((1.0 - giou).sum() / avg_factor)

        losses = {
            'loss_recon_bbox_cls': (
                torch.stack(cls_losses).sum() * loss_weights['cls_loss']),
            'loss_recon_bbox_center': (
                torch.stack(center_losses).sum()
                * loss_weights['center_loss']),
            'loss_recon_bbox_size': (
                torch.stack(size_losses).sum() * loss_weights['size_loss']),
            'loss_recon_bbox_giou': (
                torch.stack(giou_losses).sum() * loss_weights['iou_loss']),
        }
        diagnostics = {
            'num_bbox_targets': local_num_pos,
            'matches': [(pred.detach(), gt.detach()) for pred, gt in matches],
        }
        return losses, diagnostics
