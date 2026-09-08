import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import MultiScaleDeformableAttention


def inverse_sigmoid(value, eps=1e-5):
    value = value.clamp(min=0.0, max=1.0)
    value = value.clamp(min=eps, max=1.0 - eps)
    return torch.log(value / (1.0 - value))


def denormalize_reference_points(reference_points, reference_min,
                                 reference_max):
    return reference_min[:, None] + reference_points * (
        reference_max - reference_min)[:, None]


def validate_attention_blocks(attention_blocks, num_layers):
    if len(attention_blocks) != num_layers:
        raise ValueError(
            'attention_blocks must have one entry per decoder layer')
    supported = {'proj', 'bbox'}
    normalized = []
    for layer_blocks in attention_blocks:
        layer_blocks = tuple(layer_blocks)
        if not layer_blocks:
            raise ValueError(
                'Each decoder layer needs at least one attention block')
        unknown = set(layer_blocks) - supported
        if unknown:
            raise ValueError(f'Unsupported attention block: {sorted(unknown)}')
        if len(set(layer_blocks)) != len(layer_blocks):
            raise ValueError('Attention blocks cannot repeat within one layer')
        normalized.append(layer_blocks)
    return tuple(normalized)


def pack_selected_reconstruction_memory(selected_scenes):
    """Pad variable-length scene memories without changing correspondence."""
    if not selected_scenes:
        raise ValueError('selected_scenes cannot be empty')
    key_map = dict(
        points_3d='points_aligned',
        points_2d='reference_points_2d',
        bboxes_2d='bbox_preds',
        class_scores='class_scores_2d',
        view_ids='source_view_id')
    required = tuple(key_map.values()) + ('valid_mask',)
    for scene in selected_scenes:
        missing = [key for key in required if key not in scene]
        if missing:
            raise KeyError(f'Missing selected reconstruction fields: {missing}')
    max_points = max(scene['points_aligned'].shape[0]
                     for scene in selected_scenes)
    if max_points == 0:
        raise ValueError('selected reconstruction memory cannot be empty')

    packed = {}
    for output_key, source_key in key_map.items():
        values = []
        for scene in selected_scenes:
            value = scene[source_key]
            pad_shape = (max_points - value.shape[0], *value.shape[1:])
            pad = value.new_zeros(pad_shape)
            values.append(torch.cat((value, pad), dim=0))
        packed[output_key] = torch.stack(values)

    masks = []
    for scene in selected_scenes:
        count = scene['points_aligned'].shape[0]
        source_valid = scene['valid_mask'].bool()
        source_valid = source_valid.reshape(count, -1).all(dim=-1)
        masks.append(torch.cat((
            source_valid,
            source_valid.new_zeros(max_points - count)), dim=0))
    packed['valid_mask'] = torch.stack(masks)
    return packed


def _validate_memory(points_3d, points_2d, bboxes_2d, class_scores,
                     view_ids, valid_mask):
    batch_size, num_points, _ = points_3d.shape
    if points_2d.shape != (batch_size, num_points, 2):
        raise ValueError('points_2d must have shape [B, N, 2]')
    if bboxes_2d.shape != (batch_size, num_points, 4):
        raise ValueError('bboxes_2d must have shape [B, N, 4]')
    if class_scores.ndim != 3 or class_scores.shape[:2] != (
            batch_size, num_points):
        raise ValueError('class_scores must have shape [B, N, C]')
    if view_ids.shape != (batch_size, num_points):
        raise ValueError('view_ids must have shape [B, N]')
    if valid_mask.shape != (batch_size, num_points):
        raise ValueError('valid_mask must have shape [B, N]')


def _query_class_scores(class_scores, query_classes):
    batch_size, num_points, num_classes = class_scores.shape
    if query_classes.ndim != 2 or query_classes.shape[0] != batch_size:
        raise ValueError('query_classes must have shape [B, Q]')
    if ((query_classes < 0) | (query_classes >= num_classes)).any():
        raise ValueError('query_classes contains an invalid class index')
    expanded = class_scores[:, None].expand(
        -1, query_classes.shape[1], -1, -1)
    indices = query_classes[:, :, None, None].expand(
        -1, -1, num_points, 1)
    return expanded.gather(-1, indices).squeeze(-1)


def masked_view_softmax(scores, valid_mask, dim):
    if scores.shape != valid_mask.shape:
        raise ValueError('scores and valid_mask must have the same shape')
    masked_scores = scores.masked_fill(
        ~valid_mask, torch.finfo(scores.dtype).min)
    weights = masked_scores.softmax(dim=dim) * valid_mask.to(scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(
        torch.finfo(scores.dtype).eps)


def prepare_multilevel_references(reference_points, valid_ratios):
    if reference_points.ndim != 4 or reference_points.shape[-1] != 2:
        raise ValueError('reference_points must have shape [B, Q, V, 2]')
    batch_size, num_queries, num_views, _ = reference_points.shape
    if (valid_ratios.ndim != 4 or valid_ratios.shape[:2] !=
            (batch_size, num_views) or valid_ratios.shape[-1] != 2):
        raise ValueError('valid_ratios must have shape [B, V, L, 2]')
    references = reference_points.permute(0, 2, 1, 3)[:, :, :, None]
    references = references * valid_ratios[:, :, None]
    return references.reshape(
        batch_size * num_views, num_queries, valid_ratios.shape[2], 2
    ).contiguous()


def _select_references(metric, candidate_mask, points_2d, view_ids,
                       num_views, largest):
    batch_size, num_queries, _ = metric.shape
    references = points_2d.new_full(
        (batch_size, num_queries, num_views, 2), 0.5)
    view_mask = torch.zeros(
        (batch_size, num_queries, num_views), dtype=torch.bool,
        device=points_2d.device)
    fill = -torch.inf if largest else torch.inf
    for batch_id in range(batch_size):
        for view_id in range(num_views):
            point_indices = torch.nonzero(
                view_ids[batch_id] == view_id, as_tuple=False).flatten()
            if point_indices.numel() == 0:
                continue
            mask = candidate_mask[batch_id, :, point_indices]
            valid_queries = mask.any(dim=-1)
            view_mask[batch_id, :, view_id] = valid_queries
            view_metric = metric[batch_id, :, point_indices].masked_fill(
                ~mask, fill)
            selected_local = (view_metric.argmax(dim=-1) if largest
                              else view_metric.argmin(dim=-1))
            selected_global = point_indices[selected_local]
            references[batch_id, valid_queries, view_id] = points_2d[
                batch_id, selected_global[valid_queries]]
    return references, view_mask


def build_projection_memory_references(query_centers, query_classes,
                                       points_3d, points_2d, bboxes_2d,
                                       class_scores, view_ids, valid_mask,
                                       num_views):
    """Choose the nearest same-class correspondence and sample its box center."""
    _validate_memory(points_3d, points_2d, bboxes_2d, class_scores,
                     view_ids, valid_mask)
    box_center = bboxes_2d[..., :2]
    half_size = bboxes_2d[..., 2:].clamp_min(0) / 2
    inside = ((points_2d >= box_center - half_size)
              & (points_2d <= box_center + half_size)).all(dim=-1)
    candidate_labels = class_scores.argmax(dim=-1)
    same_class = candidate_labels[:, None] == query_classes[:, :, None]
    candidates = valid_mask[:, None] & inside[:, None] & same_class
    distances = torch.cdist(query_centers.float(), points_3d.float())
    return _select_references(
        distances, candidates, box_center, view_ids, num_views, largest=False)


def build_bbox_memory_references(box_centers, box_sizes, query_classes,
                                 points_3d, points_2d, bboxes_2d,
                                 class_scores, view_ids, valid_mask,
                                 num_views):
    """Select the highest query-class-score 2D point inside each 3D box."""
    _validate_memory(points_3d, points_2d, bboxes_2d, class_scores,
                     view_ids, valid_mask)
    half_size = box_sizes.clamp_min(0) / 2
    box_min = box_centers[:, :, None] - half_size[:, :, None]
    box_max = box_centers[:, :, None] + half_size[:, :, None]
    inside = ((points_3d[:, None] >= box_min)
              & (points_3d[:, None] <= box_max)).all(dim=-1)
    candidates = valid_mask[:, None] & inside
    scores = _query_class_scores(class_scores, query_classes)
    return _select_references(
        scores, candidates, points_2d, view_ids, num_views, largest=True)


class ProjectedDeformableCrossAttention(nn.Module):

    def __init__(self, embed_dims, num_heads, num_feature_levels,
                 num_points, dropout=0.0):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.deformable_attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)

    def forward(self, query, query_pos, feature_maps, reference_points,
                view_mask, valid_ratios):
        batch_size, num_queries, channels = query.shape
        num_views = reference_points.shape[2]
        if len(feature_maps) != self.num_feature_levels:
            raise ValueError(
                f'Expected {self.num_feature_levels} feature levels, '
                f'got {len(feature_maps)}')

        value_levels = []
        spatial_shapes = []
        for feature_map in feature_maps:
            _, views, feature_channels, height, width = feature_map.shape
            if views != num_views or feature_channels != channels:
                raise ValueError(
                    'Feature-map views or channels do not match the queries')
            value_levels.append(
                feature_map.permute(0, 1, 3, 4, 2).reshape(
                    batch_size * num_views, height * width, channels))
            spatial_shapes.append((height, width))

        value = torch.cat(value_levels, dim=1).contiguous()
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=query.device)
        level_start_index = torch.cat([
            spatial_shapes.new_zeros(1),
            spatial_shapes.prod(dim=1).cumsum(dim=0)[:-1]
        ])

        query_per_view = query[:, None].expand(
            -1, num_views, -1, -1).reshape(
                batch_size * num_views, num_queries, channels)
        query_pos_per_view = query_pos[:, None].expand(
            -1, num_views, -1, -1).reshape_as(query_per_view)
        reference_per_view = prepare_multilevel_references(
            reference_points.nan_to_num(0.5).clamp(0., 1.), valid_ratios)
        if reference_per_view.shape[2] != self.num_feature_levels:
            raise ValueError('valid_ratios levels must match feature levels')

        output_dtype = query.dtype
        with torch.autocast(device_type=query.device.type, enabled=False):
            attended = self.deformable_attn(
                query=query_per_view.float(),
                value=value.float(),
                identity=torch.zeros_like(
                    query_per_view, dtype=torch.float32),
                query_pos=query_pos_per_view.float(),
                reference_points=reference_per_view.float(),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)
        attended = attended.to(output_dtype).reshape(
            batch_size, num_views, num_queries, channels)

        valid_views = view_mask.permute(0, 2, 1)
        attended = attended * valid_views[..., None]
        view_query = (query + query_pos)[:, None]
        view_scores = (view_query * attended).sum(dim=-1) / math.sqrt(channels)
        view_weights = masked_view_softmax(view_scores, valid_views, dim=1)
        return (view_weights[..., None] * attended).sum(dim=1)


class GeometryAwareDecoderLayer(nn.Module):

    def __init__(self, embed_dims, num_heads, feedforward_channels,
                 num_feature_levels, attention_blocks, num_points,
                 dropout=0.0):
        super().__init__()
        self.attention_blocks = tuple(attention_blocks)
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.ModuleDict({
            block: ProjectedDeformableCrossAttention(
                embed_dims, num_heads, num_feature_levels,
                num_points[block], dropout)
            for block in self.attention_blocks
        })
        self.cross_norms = nn.ModuleDict({
            block: nn.LayerNorm(embed_dims)
            for block in self.attention_blocks
        })
        self.linear1 = nn.Linear(embed_dims, feedforward_channels)
        self.linear2 = nn.Linear(feedforward_channels, embed_dims)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, query_pos, feature_maps, block_references,
                valid_ratios):
        query_norm = self.norm1(query)
        self_attended = self.self_attn(
            query_norm + query_pos,
            query_norm + query_pos,
            query_norm,
            need_weights=False)[0]
        query = query + self.dropout(self_attended)

        for block in self.attention_blocks:
            reference_points, view_mask = block_references[block]
            cross_attended = self.cross_attn[block](
                self.cross_norms[block](query), query_pos, feature_maps,
                reference_points, view_mask, valid_ratios)
            query = query + self.dropout(cross_attended)

        ffn = self.linear2(self.dropout(F.gelu(self.linear1(self.norm3(query)))))
        return query + self.dropout(ffn)


class GeometryAwareDeformableDecoder(nn.Module):

    def __init__(self, embed_dims, num_layers, num_heads,
                 feedforward_channels, num_feature_levels,
                 attention_blocks=None, proj_num_points=4,
                 bbox_num_points=4, initial_size_anchor=(1.0, 1.0, 1.0),
                 size_logit_range=(-5.0, 5.0), dropout=0.0):
        super().__init__()
        if attention_blocks is None:
            attention_blocks = [['proj', 'bbox']] * num_layers
        self.attention_blocks = validate_attention_blocks(
            attention_blocks, num_layers)
        num_points = dict(proj=int(proj_num_points), bbox=int(bbox_num_points))
        if min(num_points.values()) <= 0:
            raise ValueError('Attention num_points must be positive')
        self.layers = nn.ModuleList([
            GeometryAwareDecoderLayer(
                embed_dims, num_heads, feedforward_channels,
                num_feature_levels, self.attention_blocks[layer_id],
                num_points, dropout)
            for layer_id in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dims)
        anchor = torch.tensor(initial_size_anchor, dtype=torch.float32)
        if anchor.shape != (3,) or (anchor <= 0).any():
            raise ValueError(
                'initial_size_anchor must contain three positive values')
        self.register_buffer('initial_size_anchor', anchor, persistent=False)
        self.size_logit_range = tuple(float(value) for value in size_logit_range)

    def forward(self, query, feature_maps, reference_points, reference_min,
                reference_max, selected_reconstruction, query_classes,
                valid_ratios, position_embedding, query_projection,
                center_branches, size_branches):
        if (len(center_branches) != len(self.layers)
                or len(size_branches) != len(self.layers)):
            raise ValueError(
                'Each decoder layer requires center and size branches')

        memory = pack_selected_reconstruction_memory(selected_reconstruction)
        memory = {key: value.to(query.device) for key, value in memory.items()}
        num_views = feature_maps[0].shape[1]
        current_center = denormalize_reference_points(
            reference_points, reference_min, reference_max)
        current_size = self.initial_size_anchor.to(query).view(1, 1, 3).expand(
            query.shape[0], query.shape[1], 3)
        current_size_log = current_size.log()

        intermediate = []
        intermediate_references = []
        for layer_id, layer in enumerate(self.layers):
            query_pos = query_projection(
                position_embedding(
                    current_center, input_range=None)).transpose(1, 2)
            with torch.no_grad(), torch.autocast(
                    device_type=query.device.type, enabled=False):
                block_references = {}
                for block in self.attention_blocks[layer_id]:
                    if block == 'proj':
                        block_references[block] = (
                            build_projection_memory_references(
                                current_center.float(), query_classes,
                                num_views=num_views, **memory))
                    else:
                        block_references[block] = build_bbox_memory_references(
                            current_center.float(), current_size.float(),
                            query_classes, num_views=num_views, **memory)
            query = layer(
                query, query_pos, feature_maps, block_references,
                valid_ratios)
            output = self.norm(query)

            center_delta = center_branches[layer_id](
                output.transpose(1, 2)).transpose(1, 2)
            new_reference_points = (
                inverse_sigmoid(reference_points) + center_delta).sigmoid()
            refined_query_xyz = denormalize_reference_points(
                new_reference_points, reference_min, reference_max)

            intermediate.append(output.transpose(1, 2))
            intermediate_references.append(refined_query_xyz)
            reference_points = new_reference_points.detach()
            current_center = refined_query_xyz.detach()

            size_residual = size_branches[layer_id](
                output.transpose(1, 2)).transpose(1, 2).float()
            predicted_size_log = current_size_log + size_residual
            predicted_size_log = predicted_size_log.clamp(
                min=self.size_logit_range[0], max=self.size_logit_range[1])
            current_size_log = predicted_size_log.detach()
            current_size = current_size_log.exp()

        return intermediate, intermediate_references
