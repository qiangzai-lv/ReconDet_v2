import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import MultiScaleDeformableAttention
from torch.utils.checkpoint import checkpoint

from mmdet.models.layers.transformer.utils import coordinate_to_encoding


@torch.no_grad()
def compute_multiview_query_correspondence(instance_embeddings,
                                           temperature=0.07):
    """Match every query to all queries in each of the other views.

    The view weights equal a global softmax over every candidate query,
    summed within each target view. Pairwise query probabilities are not
    materialized, which keeps the persistent memory cost linear in Q.
    """
    if instance_embeddings.ndim != 4:
        raise ValueError(
            'instance_embeddings must have shape [B, V, Q, C]')
    if temperature <= 0:
        raise ValueError('temperature must be positive')

    batch_size, num_views, num_queries, _ = instance_embeddings.shape
    top_indices = torch.zeros(
        (batch_size, num_views, num_queries, num_views),
        dtype=torch.long, device=instance_embeddings.device)
    view_logits = instance_embeddings.new_full(
        (batch_size, num_views, num_queries, num_views), -torch.inf,
        dtype=torch.float32)
    if num_views == 1:
        return top_indices, torch.zeros_like(view_logits)

    normalized = F.normalize(instance_embeddings.float(), dim=-1)
    for source_view in range(num_views):
        source = normalized[:, source_view]
        similarities = torch.einsum(
            'bqc,bvkc->bqvk', source, normalized)
        similarities = similarities / float(temperature)
        top_indices[:, source_view] = similarities.argmax(dim=-1)
        source_view_logits = torch.logsumexp(similarities, dim=-1)
        source_view_logits[:, :, source_view] = -torch.inf
        view_logits[:, source_view] = source_view_logits

    view_weights = torch.softmax(view_logits, dim=-1)
    view_weights = view_weights.masked_fill(
        torch.eye(num_views, dtype=torch.bool,
                  device=view_weights.device)[None, :, None], 0)
    return top_indices, view_weights


def gather_multiview_reference_points(reference_points, top_indices):
    """Gather the Top-1 target-view location for every source query."""
    if reference_points.ndim != 4 or reference_points.shape[-1] != 2:
        raise ValueError('reference_points must have shape [B, V, Q, 2]')
    batch_size, num_views, num_queries, _ = reference_points.shape
    expected_shape = (batch_size, num_views, num_queries, num_views)
    if top_indices.shape != expected_shape:
        raise ValueError(
            f'top_indices must have shape {expected_shape}, got '
            f'{tuple(top_indices.shape)}')

    gathered = reference_points.new_zeros(
        batch_size, num_views, num_queries, num_views, 2)
    for target_view in range(num_views):
        candidates = reference_points[:, target_view].unsqueeze(1).expand(
            -1, num_views, -1, -1)
        indices = top_indices[..., target_view, None].expand(-1, -1, -1, 2)
        gathered[..., target_view, :] = candidates.gather(2, indices)
    return gathered


def build_spatial_key_padding_mask(spatial_shapes, valid_ratios):
    """Build flattened per-level padding masks from valid image ratios."""
    if spatial_shapes.ndim != 2 or spatial_shapes.shape[-1] != 2:
        raise ValueError('spatial_shapes must have shape [L, 2]')
    if (valid_ratios.ndim != 3 or valid_ratios.shape[1] != len(spatial_shapes)
            or valid_ratios.shape[-1] != 2):
        raise ValueError('valid_ratios must have shape [N, L, 2]')

    level_masks = []
    for level, (height, width) in enumerate(spatial_shapes.tolist()):
        valid_width = torch.round(
            valid_ratios[:, level, 0] * width).long().clamp(0, width)
        valid_height = torch.round(
            valid_ratios[:, level, 1] * height).long().clamp(0, height)
        rows = torch.arange(height, device=valid_ratios.device)[None, :, None]
        columns = torch.arange(width, device=valid_ratios.device)[None, None]
        mask = ((rows >= valid_height[:, None, None]) |
                (columns >= valid_width[:, None, None]))
        level_masks.append(mask.flatten(1))
    return torch.cat(level_masks, dim=1)


class FourierPositionEmbedding2D(nn.Module):

    def __init__(self, embed_dims, num_bands=8):
        super().__init__()
        self.register_buffer(
            'frequencies',
            2.0 ** torch.arange(num_bands, dtype=torch.float32) * torch.pi,
            persistent=False)
        fourier_dims = 2 + 2 * num_bands * 2
        self.projection = nn.Linear(fourier_dims, embed_dims)

    def forward(self, reference_points):
        uv = reference_points.float()
        scaled = uv.unsqueeze(-1) * self.frequencies.view(1, 1, 1, -1)
        encoded = torch.cat(
            [uv.unsqueeze(-1), scaled.sin(), scaled.cos()], dim=-1)
        encoded = encoded.flatten(start_dim=2)
        return self.projection(encoded)


def recover_feature_maps(memory, spatial_shapes):
    feature_maps = []
    start = 0
    for height, width in spatial_shapes.tolist():
        length = height * width
        level_memory = memory[:, start:start + length]
        feature_maps.append(
            level_memory.transpose(1, 2).reshape(
                memory.shape[0], memory.shape[-1], height, width).contiguous())
        start += length
    return feature_maps


def flatten_feature_maps(feature_maps):
    values = []
    spatial_shapes = []
    for feature_map in feature_maps:
        batch_size, channels, height, width = feature_map.shape
        values.append(
            feature_map.flatten(2).transpose(1, 2).contiguous())
        spatial_shapes.append((height, width))

    value = torch.cat(values, dim=1)
    spatial_shapes = torch.as_tensor(
        spatial_shapes, dtype=torch.long, device=value.device)
    level_start_index = torch.cat([
        spatial_shapes.new_zeros(1),
        spatial_shapes.prod(dim=1).cumsum(dim=0)[:-1]
    ])
    return value, spatial_shapes, level_start_index


class ProjectedQueryDeformableAttention(nn.Module):

    def __init__(self, query_dims, value_dims, num_heads, num_levels,
                 num_points, dropout=0.0):
        super().__init__()
        self.num_levels = num_levels
        self.query_projection = nn.Linear(query_dims, value_dims)
        self.position_projection = nn.Linear(query_dims, value_dims)
        self.output_projection = nn.Linear(value_dims, query_dims)
        self.attention = MultiScaleDeformableAttention(
            embed_dims=value_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)

    def forward(self, query, query_pos, value, spatial_shapes,
                level_start_index, reference_points, valid_ratios,
                key_padding_mask=None):
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        projected_query = self.query_projection(query)
        projected_position = self.position_projection(query_pos)

        output_dtype = query.dtype
        with torch.autocast(device_type=query.device.type, enabled=False):
            attended = self.attention(
                query=projected_query.float(),
                value=value.float(),
                identity=torch.zeros_like(projected_query, dtype=torch.float32),
                query_pos=projected_position.float(),
                key_padding_mask=key_padding_mask,
                reference_points=reference_points.float(),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)
        return self.output_projection(attended.to(output_dtype))


class GroundingDINO3DDecoderLayer(nn.Module):

    def __init__(self, query_dims, spatial_dims, num_heads,
                 feedforward_channels, num_feature_levels, num_points,
                 dropout=0.0):
        super().__init__()
        self.spatial_attention = ProjectedQueryDeformableAttention(
            query_dims=query_dims,
            value_dims=spatial_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            dropout=dropout)
        self.cross_view_attention = ProjectedQueryDeformableAttention(
            query_dims=query_dims,
            value_dims=spatial_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            dropout=dropout)
        self.spatial_norm = nn.LayerNorm(query_dims)
        self.cross_view_norm = nn.LayerNorm(query_dims)
        self.cross_view_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.ffn_norm = nn.LayerNorm(query_dims)
        self.linear1 = nn.Linear(query_dims, feedforward_channels)
        self.linear2 = nn.Linear(feedforward_channels, query_dims)
        self.dropout = nn.Dropout(dropout)

    def _cross_view_forward(self, query, query_pos, spatial_features,
                            valid_ratios, spatial_key_padding_mask, num_views,
                            cross_reference_points, cross_view_weights,
                            cross_query_positions=None):
        spatial_value, spatial_shapes, spatial_level_start = spatial_features
        num_flat_views, num_queries, _ = query.shape
        if num_views <= 1:
            return torch.zeros_like(query)
        if num_flat_views % num_views != 0:
            raise ValueError('flattened views must be divisible by num_views')
        batch_size = num_flat_views // num_views
        expected_references = (
            batch_size, num_views, num_queries, num_views, 2)
        expected_weights = expected_references[:-1]
        if cross_reference_points.shape != expected_references:
            raise ValueError(
                'cross_reference_points must have shape '
                f'{expected_references}')
        if cross_view_weights.shape != expected_weights:
            raise ValueError(
                f'cross_view_weights must have shape {expected_weights}')
        expected_positions = expected_references[:-1] + (query.shape[-1],)
        if cross_query_positions is None:
            cross_query_positions = query_pos.reshape(
                batch_size, num_views, num_queries, 1, -1).expand(
                    expected_positions)
        elif cross_query_positions.shape != expected_positions:
            raise ValueError(
                f'cross_query_positions must have shape {expected_positions}')

        _, sequence_length, spatial_dims = spatial_value.shape
        num_levels = valid_ratios.shape[1]
        values_by_view = spatial_value.reshape(
            batch_size, num_views, sequence_length, spatial_dims)
        ratios_by_view = valid_ratios.reshape(
            batch_size, num_views, num_levels, 2)
        masks_by_view = spatial_key_padding_mask.reshape(
            batch_size, num_views, sequence_length)
        normalized_query = self.cross_view_norm(query)
        cross_view_output = torch.zeros_like(query)

        def attend_target_view(query_input, position_input, target_value_base,
                               target_references, target_ratio_base,
                               target_mask_base):
            target_value = target_value_base.unsqueeze(1).expand(
                -1, num_views, -1, -1).reshape(
                    num_flat_views, sequence_length, spatial_dims)
            target_ratios = target_ratio_base.unsqueeze(1).expand(
                -1, num_views, -1, -1).reshape(
                    num_flat_views, num_levels, 2)
            target_mask = target_mask_base.unsqueeze(1).expand(
                -1, num_views, -1).reshape(num_flat_views, sequence_length)
            return self.cross_view_attention(
                query_input, position_input, target_value,
                spatial_shapes, spatial_level_start, target_references,
                target_ratios, key_padding_mask=target_mask)

        for target_view in range(num_views):
            target_references = cross_reference_points[
                ..., target_view, :].reshape(num_flat_views, num_queries, 2)
            target_positions = cross_query_positions[
                ..., target_view, :].reshape_as(query_pos)
            attention_inputs = (
                normalized_query, target_positions,
                values_by_view[:, target_view],
                target_references, ratios_by_view[:, target_view],
                masks_by_view[:, target_view])
            if (self.training and torch.is_grad_enabled() and
                    (query.requires_grad or spatial_value.requires_grad)):
                attended = checkpoint(
                    attend_target_view, *attention_inputs,
                    use_reentrant=False)
            else:
                attended = attend_target_view(*attention_inputs)
            target_weights = cross_view_weights[
                ..., target_view].reshape(
                    num_flat_views, num_queries, 1).to(attended.dtype)
            cross_view_output.add_(attended * target_weights)
        return cross_view_output

    def forward(self, query, query_pos, spatial_features, reference_points,
                valid_ratios, num_views=1, cross_reference_points=None,
                cross_view_weights=None, cross_query_positions=None,
                spatial_key_padding_mask=None):
        spatial_value, spatial_shapes, spatial_level_start = spatial_features
        if spatial_key_padding_mask is None:
            spatial_key_padding_mask = build_spatial_key_padding_mask(
                spatial_shapes, valid_ratios)
        spatial = self.spatial_attention(
            self.spatial_norm(query), query_pos, spatial_value,
            spatial_shapes, spatial_level_start, reference_points,
            valid_ratios, key_padding_mask=spatial_key_padding_mask)
        query = query + self.dropout(spatial)

        if ((cross_reference_points is None) !=
                (cross_view_weights is None)):
            raise ValueError(
                'cross-view references and weights must be provided together')
        if cross_reference_points is not None and num_views > 1:
            cross_view = self._cross_view_forward(
                query, query_pos, spatial_features, valid_ratios,
                spatial_key_padding_mask, num_views, cross_reference_points,
                cross_view_weights, cross_query_positions)
            query = query + self.dropout(
                self.cross_view_residual_scale * cross_view)

        ffn = self.linear2(self.dropout(F.gelu(
            self.linear1(self.ffn_norm(query)))))
        return query + self.dropout(ffn)


class GroundingDINO3DDecoder(nn.Module):

    def __init__(self, num_queries, query_dims=512, semantic_dims=256,
                 spatial_dims=512, num_layers=6, num_heads=8,
                 feedforward_channels=2048, num_feature_levels=4,
                 num_points=4, dropout=0.0, camera_dims=2048,
                 uv_num_bands=8):
        super().__init__()
        self.camera_dims = int(camera_dims)
        self.num_queries = num_queries
        self.semantic_query_projection = nn.Linear(semantic_dims, query_dims)
        self.query_embedding = nn.Embedding(num_queries, query_dims)
        self.init_norm = nn.LayerNorm(query_dims)
        self.query_position_embedding = FourierPositionEmbedding2D(
            query_dims, num_bands=uv_num_bands)
        self.camera_init_projection = nn.Sequential(
            nn.Linear(camera_dims, camera_dims // 2), nn.GELU(),
            nn.Linear(camera_dims // 2, query_dims))
        self.reference_projection = nn.Sequential(
            nn.Linear(semantic_dims, query_dims),
            nn.ReLU(),
            nn.Linear(query_dims, query_dims))
        self.layers = nn.ModuleList([
            GroundingDINO3DDecoderLayer(
                query_dims=query_dims,
                spatial_dims=spatial_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                num_feature_levels=num_feature_levels,
                num_points=num_points,
                dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(query_dims)
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            layer.spatial_attention.attention.init_weights()
            layer.cross_view_attention.attention.init_weights()

    def initialize_query(self, semantic_query, reference_points, camera_tokens):
        if semantic_query.ndim != 3:
            raise ValueError('semantic_query must have shape [N, Q, C]')
        if semantic_query.shape[1] != self.num_queries:
            raise ValueError(
                'semantic query count must match reconstruction query count')
        if reference_points.shape != semantic_query.shape[:2] + (2,):
            raise ValueError(
                'reference_points must have shape [N, Q, 2] matching query')
        if camera_tokens is None or camera_tokens.shape != (
                semantic_query.shape[0], self.camera_dims):
            raise ValueError('camera_tokens must have shape [B*V, camera_dims]')
        learned_query = self.query_embedding.weight[None].expand(
            semantic_query.shape[0], -1, -1)
        projected_semantic = self.semantic_query_projection(semantic_query)
        projected_uv = self.query_position_embedding(reference_points)
        projected_camera = self.camera_init_projection(camera_tokens)
        return self.init_norm(
            learned_query + projected_semantic + projected_uv
            + projected_camera[:, None, :])

    def forward(self, semantic_query, spatial_features, reference_points,
                valid_ratios, instance_embeddings=None, num_views=1,
                correspondence_temperature=0.07, camera_tokens=None):
        if num_views <= 0:
            raise ValueError('num_views must be positive')
        if semantic_query.shape[0] % num_views != 0:
            raise ValueError(
                'semantic query batch must be divisible by num_views')

        cross_reference_points = None
        cross_view_weights = None
        cross_query_positions = None
        if num_views > 1:
            if instance_embeddings is None:
                raise ValueError(
                    'instance_embeddings are required for multiview decoding')
            if instance_embeddings.shape[:2] != semantic_query.shape[:2]:
                raise ValueError(
                    'instance_embeddings must begin with [B*V, Q]')
            batch_size = semantic_query.shape[0] // num_views
            embeddings_by_scene = instance_embeddings.reshape(
                batch_size, num_views, self.num_queries, -1)
            references_by_scene = reference_points.reshape(
                batch_size, num_views, self.num_queries, 2)
            top_indices, cross_view_weights = (
                compute_multiview_query_correspondence(
                    embeddings_by_scene,
                    temperature=correspondence_temperature))
            cross_reference_points = gather_multiview_reference_points(
                references_by_scene, top_indices)
            flat_cross_references = cross_reference_points.reshape(
                semantic_query.shape[0], self.num_queries * num_views, 2)
            cross_query_positions = self.reference_projection(
                coordinate_to_encoding(flat_cross_references)).reshape(
                    batch_size, num_views, self.num_queries, num_views, -1)

        query = self.initialize_query(
            semantic_query, reference_points, camera_tokens)
        query_pos = self.reference_projection(
            coordinate_to_encoding(reference_points))
        spatial_key_padding_mask = build_spatial_key_padding_mask(
            spatial_features[1], valid_ratios)
        intermediate = []
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                spatial_features=spatial_features,
                reference_points=reference_points,
                valid_ratios=valid_ratios,
                num_views=num_views,
                cross_reference_points=cross_reference_points,
                cross_view_weights=cross_view_weights,
                cross_query_positions=cross_query_positions,
                spatial_key_padding_mask=spatial_key_padding_mask)
            intermediate.append(self.norm(query))
        return torch.stack(intermediate)
