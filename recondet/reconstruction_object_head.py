from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet3d.structures.ops.iou3d_calculator import (
    axis_aligned_bbox_overlaps_3d)


def group_scene_instances(scene_indices: Tensor, instance_ids: Tensor):
    """Assign flattened points to scene-local instances."""
    if scene_indices.ndim != 1 or instance_ids.shape != scene_indices.shape:
        raise ValueError('scene_indices and instance_ids must have shape [M]')
    keys = torch.stack([scene_indices.long(), instance_ids.long()], dim=-1)
    unique_keys, group_indices = torch.unique(
        keys, dim=0, sorted=True, return_inverse=True)
    counts = torch.bincount(
        group_indices, minlength=len(unique_keys))
    return unique_keys, group_indices, counts


@torch.no_grad()
def group_instance_embeddings(embeddings: Tensor, scene_indices: Tensor,
                              view_indices: Tensor,
                              similarity_threshold: float = 0.7) -> Tensor:
    """Group reconstruction queries by cross-view embedding affinity."""
    if embeddings.ndim != 2:
        raise ValueError('embeddings must have shape [M, C]')
    if scene_indices.shape != (len(embeddings),):
        raise ValueError('scene_indices must have shape [M]')
    if view_indices.shape != (len(embeddings),):
        raise ValueError('view_indices must have shape [M]')
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError('similarity_threshold must be between 0 and 1')
    if len(embeddings) == 0:
        return torch.empty(0, dtype=torch.long, device=embeddings.device)

    if not torch.isfinite(embeddings).all():
        raise ValueError('embeddings must contain only finite values')
    normalized = F.normalize(embeddings.float(), dim=-1)
    scene_indices = scene_indices.detach().cpu().tolist()
    view_indices = view_indices.detach().cpu().tolist()
    group_members = []
    group_scenes = []
    group_views = []
    assignments = []
    for index in range(len(embeddings)):
        best_group = -1
        best_similarity = similarity_threshold
        for group_id, members in enumerate(group_members):
            if group_scenes[group_id] != scene_indices[index]:
                continue
            if view_indices[index] in group_views[group_id]:
                continue
            prototype = normalized[members].mean(dim=0)
            similarity = torch.dot(normalized[index], F.normalize(
                prototype, dim=0)).item()
            if similarity >= best_similarity:
                best_group = group_id
                best_similarity = similarity
        if best_group < 0:
            best_group = len(group_members)
            group_members.append([])
            group_scenes.append(scene_indices[index])
            group_views.append(set())
        group_members[best_group].append(index)
        group_views[best_group].add(view_indices[index])
        assignments.append(best_group)
    return torch.tensor(assignments, dtype=torch.long, device=embeddings.device)


def _group_mean(values: Tensor, group_indices: Tensor,
                num_groups: int) -> Tensor:
    output = values.new_zeros((num_groups,) + values.shape[1:])
    output.index_add_(0, group_indices, values)
    counts = torch.bincount(
        group_indices, minlength=num_groups).to(values).clamp_min(1)
    return output / counts.reshape(
        num_groups, *([1] * (values.ndim - 1)))


def _group_max(values: Tensor, group_indices: Tensor,
               num_groups: int) -> Tensor:
    output = values.new_full((num_groups,) + values.shape[1:], -torch.inf)
    expanded_indices = group_indices.reshape(
        -1, *([1] * (values.ndim - 1))).expand_as(values)
    output.scatter_reduce_(
        0, expanded_indices, values, reduce='amax', include_self=True)
    return output


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


def aggregate_point_set_boxes(
        points: Tensor, center_offsets: Tensor, completion_logits: Tensor,
        group_indices: Tensor, num_groups: int):
    """Aggregate point votes and complete the observed symmetric extent."""
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError('points must have shape [M, 3]')
    if center_offsets.shape != points.shape:
        raise ValueError('center_offsets must match points')
    if completion_logits.shape != (num_groups, 3):
        raise ValueError('completion_logits must have shape [G, 3]')
    if group_indices.shape != (len(points),):
        raise ValueError('group_indices must have shape [M]')

    center_votes = points + center_offsets
    centers = _group_mean(center_votes, group_indices, num_groups)
    relative_points = points - centers[group_indices]
    observed_half_sizes = _group_max(
        relative_points.abs(), group_indices, num_groups)
    half_sizes = observed_half_sizes + F.softplus(completion_logits.float())
    return centers, 2.0 * half_sizes, observed_half_sizes


def _center_size_to_minmax(centers: Tensor, sizes: Tensor) -> Tensor:
    half_sizes = sizes / 2.0
    return torch.cat([centers - half_sizes, centers + half_sizes], dim=-1)


class FourierPositionEncoding(nn.Module):

    def __init__(self, num_frequencies: int = 4):
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError('num_frequencies must be positive')
        frequencies = 2.0 ** torch.arange(num_frequencies).float()
        self.register_buffer(
            'frequencies', frequencies, persistent=False)
        self.output_dims = 3 * (1 + 2 * num_frequencies)

    def forward(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
            raise ValueError('coordinates must have shape [M, 3]')
        angles = (
            coordinates.float()[..., None]
            * self.frequencies * torch.pi)
        return torch.cat([
            coordinates.float(),
            angles.sin().flatten(start_dim=-2),
            angles.cos().flatten(start_dim=-2),
        ], dim=-1)


def _empty_object_samples(outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    queries = outputs['reconstruction_query']
    points = outputs['points_aligned']
    return {
        'queries': queries.new_empty((0, queries.shape[-1])),
        'points': points.new_empty((0, 3)),
        'surface_targets': points.new_empty((0, 3)),
        'box_centers': points.new_empty((0, 3)),
        'box_sizes': points.new_empty((0, 3)),
        'scene_indices': torch.empty(
            0, dtype=torch.long, device=queries.device),
        'view_indices': torch.empty(
            0, dtype=torch.long, device=queries.device),
        'instance_ids': torch.empty(
            0, dtype=torch.long, device=queries.device),
    }


def collect_matched_object_samples(
        reconstruction_outputs: Dict[str, Tensor],
        matches: List[Tuple[Tensor, Tensor]], batch_data_samples,
        num_views: int) -> Dict[str, Tensor]:
    """Join matched 2D queries to aligned surface points and ScanNet boxes."""
    required = {
        'reconstruction_query', 'points_aligned', 'valid_mask'}
    missing = required - set(reconstruction_outputs)
    if missing:
        raise KeyError(f'missing reconstruction outputs: {sorted(missing)}')
    queries = reconstruction_outputs['reconstruction_query']
    points = reconstruction_outputs['points_aligned']
    valid_mask = reconstruction_outputs['valid_mask']
    if num_views <= 0:
        raise ValueError('num_views must be positive')
    expected_views = len(batch_data_samples) * num_views
    if len(matches) != expected_views or queries.shape[:2] != points.shape[:2]:
        raise ValueError('matches and reconstruction outputs must share B*V')
    if queries.shape[0] != expected_views or valid_mask.shape != points.shape[:2]:
        raise ValueError('reconstruction outputs must use flattened B*V views')

    collected = {name: [] for name in (
        'queries', 'points', 'surface_targets', 'box_centers', 'box_sizes',
        'scene_indices', 'view_indices', 'instance_ids')}
    for flat_view, (query_indices, gt_indices) in enumerate(matches):
        if query_indices.numel() == 0:
            continue
        scene_index = flat_view // num_views
        view_index = flat_view % num_views
        scene_sample = batch_data_samples[scene_index]
        instances_2d = scene_sample.gt_instances_2d[view_index]
        instances_3d = scene_sample.gt_instances_3d
        if not hasattr(instances_3d, 'instance_ids_3d'):
            raise RuntimeError(
                '3D GT must contain instance_ids_3d for object supervision')
        if not hasattr(instances_2d, 'centers_3d_aligned'):
            raise RuntimeError(
                '2D GT must contain centers_3d_aligned for object supervision')

        device = queries.device
        query_indices = query_indices.to(device=device, dtype=torch.long)
        gt_indices = gt_indices.to(device=device, dtype=torch.long)
        instance_ids = instances_2d.instance_ids_3d.to(device)[gt_indices]
        box_ids = torch.as_tensor(
            instances_3d.instance_ids_3d,
            device=device, dtype=torch.long)
        id_to_box = {int(value): index
                     for index, value in enumerate(box_ids.tolist())}
        box_indices = torch.tensor(
            [id_to_box.get(int(value), -1) for value in instance_ids.tolist()],
            dtype=torch.long, device=device)

        surface_targets = instances_2d.centers_3d_aligned.to(
            device=device, dtype=torch.float32)[gt_indices]
        target_valid = instances_2d.center_3d_valid_mask.to(
            device=device, dtype=torch.bool)[gt_indices]
        predicted_points = points[flat_view, query_indices].float()
        target_valid &= valid_mask[flat_view, query_indices].bool()
        target_valid &= box_indices >= 0
        target_valid &= torch.isfinite(predicted_points).all(dim=-1)
        target_valid &= torch.isfinite(surface_targets).all(dim=-1)
        if not target_valid.any():
            continue

        query_indices = query_indices[target_valid]
        instance_ids = instance_ids[target_valid].long()
        box_indices = box_indices[target_valid]
        predicted_points = predicted_points[target_valid]
        surface_targets = surface_targets[target_valid]
        box_centers = instances_3d.bboxes_3d.gravity_center.to(
            device=device, dtype=torch.float32)[box_indices]
        box_sizes = instances_3d.bboxes_3d.tensor[:, 3:6].to(
            device=device, dtype=torch.float32)[box_indices]
        finite_boxes = torch.isfinite(box_centers).all(dim=-1)
        finite_boxes &= torch.isfinite(box_sizes).all(dim=-1)
        finite_boxes &= (box_sizes > 0).all(dim=-1)
        if not finite_boxes.any():
            continue

        count = int(finite_boxes.sum())
        collected['queries'].append(
            queries[flat_view, query_indices][finite_boxes])
        collected['points'].append(predicted_points[finite_boxes])
        collected['surface_targets'].append(surface_targets[finite_boxes])
        collected['box_centers'].append(box_centers[finite_boxes])
        collected['box_sizes'].append(box_sizes[finite_boxes])
        collected['scene_indices'].append(torch.full(
            (count,), scene_index, dtype=torch.long, device=device))
        collected['view_indices'].append(torch.full(
            (count,), view_index, dtype=torch.long, device=device))
        collected['instance_ids'].append(instance_ids[finite_boxes])

    if not collected['queries']:
        return _empty_object_samples(reconstruction_outputs)
    return {name: torch.cat(values, dim=0)
            for name, values in collected.items()}


class ReconstructionObjectHead(nn.Module):
    """Object-level supervision for variable-size reconstruction point sets."""

    def __init__(self, query_dims: int, hidden_dims: int = 256,
                 instance_dims: int = 128, temperature: float = 0.07,
                 instance_weight: float = 0.2, center_weight: float = 0.5,
                 bbox_weight: float = 1.0, giou_weight: float = 0.5,
                 min_bbox_views: int = 2, geometry_frequencies: int = 4,
                 grouping_similarity_threshold: float = 0.7):
        super().__init__()
        if min(query_dims, hidden_dims, instance_dims, min_bbox_views) <= 0:
            raise ValueError('head dimensions and min_bbox_views must be positive')
        self.temperature = float(temperature)
        self.instance_weight = float(instance_weight)
        self.center_weight = float(center_weight)
        self.bbox_weight = float(bbox_weight)
        self.giou_weight = float(giou_weight)
        self.min_bbox_views = int(min_bbox_views)
        if not 0.0 <= grouping_similarity_threshold <= 1.0:
            raise ValueError(
                'grouping_similarity_threshold must be between 0 and 1')
        self.grouping_similarity_threshold = float(
            grouping_similarity_threshold)

        self.instance_projection = nn.Sequential(
            nn.LayerNorm(query_dims), nn.Linear(query_dims, hidden_dims),
            nn.GELU(), nn.Linear(hidden_dims, instance_dims))
        self.center_offset_head = nn.Sequential(
            nn.LayerNorm(query_dims),
            nn.Linear(query_dims, hidden_dims), nn.GELU(),
            nn.Linear(hidden_dims, 3))
        self.size_query_encoder = nn.Sequential(
            nn.LayerNorm(query_dims),
            nn.Linear(query_dims, hidden_dims), nn.GELU())
        self.geometry_position_encoding = FourierPositionEncoding(
            geometry_frequencies)
        self.size_geometry_encoder = nn.Sequential(
            nn.Linear(
                self.geometry_position_encoding.output_dims, hidden_dims),
            nn.GELU(), nn.Linear(hidden_dims, hidden_dims))
        self.size_fusion_norm = nn.LayerNorm(hidden_dims)
        self.size_completion_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dims),
            nn.Linear(2 * hidden_dims, hidden_dims), nn.GELU(),
            nn.Linear(hidden_dims, 3))

    def forward(self, queries: Tensor, points: Tensor, group_indices: Tensor,
                num_groups: int) -> Dict[str, Tensor]:
        if queries.ndim != 2 or points.shape != (len(queries), 3):
            raise ValueError('queries must have shape [M, C] matching points')
        if group_indices.shape != (len(queries),):
            raise ValueError('group_indices must have shape [M]')
        if num_groups <= 0 or (len(group_indices) and
                               (group_indices.min() < 0 or
                                group_indices.max() >= num_groups)):
            raise ValueError('group_indices must be within num_groups')
        if not torch.isfinite(queries).all() or not torch.isfinite(points).all():
            raise ValueError('queries and points must contain finite values')
        queries_fp32 = queries.float()
        center_offsets = self.center_offset_head(queries_fp32).float()
        center_votes = points.float() + center_offsets
        centers = _group_mean(center_votes, group_indices, num_groups)
        relative = points.float() - centers[group_indices]
        query_features = self.size_query_encoder(queries_fp32)
        geometry_features = self.size_geometry_encoder(
            self.geometry_position_encoding(relative))
        size_features = self.size_fusion_norm(
            query_features + geometry_features)
        pooled = torch.cat([
            _group_mean(size_features, group_indices, num_groups),
            _group_max(size_features, group_indices, num_groups),
        ], dim=-1)
        completion_logits = self.size_completion_head(pooled).float()
        centers, sizes, observed = aggregate_point_set_boxes(
            points.float(), center_offsets, completion_logits,
            group_indices, num_groups)
        boxes = _center_size_to_minmax(centers, sizes)
        return {
            'center_offsets': center_offsets,
            'center_votes': center_votes,
            'centers': centers,
            'sizes': sizes,
            'boxes': boxes,
            'observed_half_sizes': observed,
            'group_queries': _group_mean(
                queries_fp32, group_indices, num_groups),
            'instance_embeddings': self.instance_projection(queries_fp32),
        }

    @torch.no_grad()
    def predict(self, samples: Dict[str, Tensor], group_indices: Tensor = None):
        """Generate object boxes directly from reconstruction queries.

        Training uses GT instance groups in :meth:`loss`; prediction uses
        cross-view instance embedding affinity and never requires 3D labels.
        """
        required = {'queries', 'points', 'scene_indices', 'view_indices'}
        missing = required - set(samples)
        if missing:
            raise KeyError(f'missing prediction samples: {sorted(missing)}')
        queries = samples['queries']
        points = samples['points']
        scene_indices = samples['scene_indices']
        view_indices = samples['view_indices']
        if queries.ndim != 2 or points.shape != (len(queries), 3):
            raise ValueError('queries and points must have shapes [M, D] and [M, 3]')
        if not torch.isfinite(queries).all() or not torch.isfinite(points).all():
            raise ValueError('queries and points must contain finite values')
        if scene_indices.shape != (len(queries),) or view_indices.shape != (len(queries),):
            raise ValueError('scene_indices and view_indices must have shape [M]')
        if scene_indices.device != queries.device or view_indices.device != queries.device:
            raise ValueError('scene_indices and view_indices must be on query device')
        if not torch.isfinite(scene_indices.float()).all() or not torch.isfinite(view_indices.float()).all():
            raise ValueError('scene_indices and view_indices must be finite')
        scene_indices = scene_indices.long()
        view_indices = view_indices.long()
        if group_indices is not None:
            if group_indices.shape != (len(queries),):
                raise ValueError('group_indices must have shape [M]')
            group_indices = group_indices.to(device=queries.device, dtype=torch.long)

        embeddings = samples.get('instance_embeddings')
        if embeddings is None:
            embeddings = self.instance_projection(queries.float())
        if embeddings.shape != (len(queries), self.instance_projection[-1].out_features):
            raise ValueError('instance_embeddings must have shape [M, instance_dims]')
        if embeddings.device != queries.device or not torch.isfinite(embeddings).all():
            raise ValueError('instance_embeddings must be finite and on query device')
        semantic_queries = samples.get('semantic_queries')
        if semantic_queries is not None:
            if semantic_queries.ndim != 2 or semantic_queries.shape[0] != len(queries):
                raise ValueError('semantic_queries must have shape [M, C]')
            if semantic_queries.device != queries.device or not torch.isfinite(semantic_queries).all():
                raise ValueError('semantic_queries must be finite and on query device')
        class_scores = samples.get('class_scores')
        if class_scores is not None:
            if class_scores.ndim != 2 or class_scores.shape[0] != len(queries):
                raise ValueError('class_scores must have shape [M, C]')
            if class_scores.device != queries.device or not torch.isfinite(class_scores).all():
                raise ValueError('class_scores must be finite and on query device')
        foreground = samples.get('foreground_scores')
        if foreground is None:
            foreground = (class_scores.amax(dim=-1)
                          if class_scores is not None else
                          points.new_ones(len(points)))
        else:
            if foreground.device != queries.device:
                raise ValueError('foreground_scores must be finite and on query device')
            foreground = foreground.to(dtype=torch.float32)
        if foreground.shape != (len(queries),):
            raise ValueError('foreground_scores must have shape [M]')
        if foreground.device != queries.device or not torch.isfinite(foreground).all():
            raise ValueError('foreground_scores must be finite and on query device')

        outputs = []
        for scene in torch.unique(scene_indices, sorted=True):
            selected = scene_indices == scene
            local_ids = torch.nonzero(selected, as_tuple=False).flatten()
            if group_indices is None:
                local_groups = group_instance_embeddings(
                    embeddings[local_ids], scene_indices[local_ids],
                    view_indices[local_ids], self.grouping_similarity_threshold)
            else:
                _, local_groups = torch.unique(
                    group_indices[local_ids], sorted=True, return_inverse=True)
            result = self(
                queries[local_ids], points[local_ids], local_groups,
                int(local_groups.max().item()) + 1)
            if semantic_queries is None:
                pooled_semantic = None
            else:
                pooled_semantic = _group_mean(
                    semantic_queries[local_ids].float(), local_groups,
                    result['boxes'].shape[0])
            group_scores = []
            group_labels = []
            for group_id in range(result['boxes'].shape[0]):
                members = local_groups == group_id
                weights = foreground[local_ids][members].float().clamp_min(1e-6)
                if class_scores is None:
                    group_scores.append(weights.mean())
                    group_labels.append(torch.tensor(
                        -1, dtype=torch.long, device=queries.device))
                else:
                    scores = class_scores[local_ids][members].float()
                    pooled = (scores * weights[:, None]).sum(0) / weights.sum()
                    group_scores.append(pooled.max())
                    group_labels.append(pooled.argmax())
            outputs.append({
                'boxes': result['boxes'],
                'scores': torch.stack(group_scores),
                'labels': torch.stack(group_labels),
                'centers': result['centers'],
                'sizes': result['sizes'],
                'observed_half_sizes': result['observed_half_sizes'],
                'instance_embeddings': result['instance_embeddings'],
                'grouping_embeddings': embeddings[local_ids],
                'group_queries': result['group_queries'],
                'semantic_queries': pooled_semantic,
                'group_indices': local_groups,
                'query_indices': local_ids,
            })
        return outputs

    def loss(self, samples: Dict[str, Tensor]):
        queries = samples['queries']
        if len(queries) == 0:
            zero = queries.sum() * 0.0
            losses = {
                'loss_recon_instance': zero,
                'loss_recon_center': zero,
                'loss_recon_bbox': zero,
            }
            return losses, {'num_points': 0, 'num_bbox_groups': 0}

        keys, group_indices, counts = group_scene_instances(
            samples['scene_indices'], samples['instance_ids'])
        predictions = self(
            queries, samples['points'], group_indices, len(keys))
        instance_loss = supervised_instance_contrastive_loss(
            predictions['instance_embeddings'], keys[group_indices],
            samples['view_indices'], self.temperature)

        target_offsets = samples['box_centers'] - samples['surface_targets']
        normalized_error = (
            predictions['center_offsets'] - target_offsets
        ) / samples['box_sizes'].clamp_min(1e-5)
        center_loss = F.smooth_l1_loss(
            normalized_error, torch.zeros_like(normalized_error),
            reduction='none').mean(dim=-1).mean()

        target_centers = _group_mean(
            samples['box_centers'], group_indices, len(keys))
        target_sizes = _group_mean(
            samples['box_sizes'], group_indices, len(keys)).clamp_min(1e-5)
        group_view_keys = torch.stack(
            [group_indices, samples['view_indices'].long()], dim=-1)
        unique_group_views = torch.unique(group_view_keys, dim=0)
        view_counts = torch.bincount(
            unique_group_views[:, 0], minlength=len(keys))
        bbox_valid = (counts >= self.min_bbox_views)
        bbox_valid &= view_counts >= self.min_bbox_views
        if bbox_valid.any():
            pred_centers = predictions['centers'][bbox_valid]
            pred_sizes = predictions['sizes'][bbox_valid].clamp_min(1e-5)
            gt_centers = target_centers[bbox_valid]
            gt_sizes = target_sizes[bbox_valid]
            size_error = torch.log(pred_sizes / gt_sizes)
            size_loss = F.smooth_l1_loss(
                size_error, torch.zeros_like(size_error),
                reduction='none').mean(dim=-1)
            pred_boxes = _center_size_to_minmax(pred_centers, pred_sizes)
            gt_boxes = _center_size_to_minmax(gt_centers, gt_sizes)
            giou = axis_aligned_bbox_overlaps_3d(
                pred_boxes.unsqueeze(0), gt_boxes.unsqueeze(0),
                mode='giou', is_aligned=True).squeeze(0)
            bbox_loss = (size_loss + self.giou_weight * (1.0 - giou)).mean()
        else:
            bbox_loss = predictions['sizes'].sum() * 0.0

        losses = {
            'loss_recon_instance': instance_loss * self.instance_weight,
            'loss_recon_center': center_loss * self.center_weight,
            'loss_recon_bbox': bbox_loss * self.bbox_weight,
        }
        diagnostics = {
            'num_points': len(queries),
            'num_bbox_groups': int(bbox_valid.sum()),
        }
        return losses, diagnostics
