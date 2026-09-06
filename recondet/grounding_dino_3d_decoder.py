import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import MultiScaleDeformableAttention

from mmdet.models.layers.transformer.utils import coordinate_to_encoding


class FourierPositionEmbedding2D(nn.Module):

    def __init__(self, embed_dims, num_bands=8):
        super().__init__()
        self.register_buffer(
            'frequencies',
            2.0 ** torch.arange(num_bands, dtype=torch.float32) * torch.pi,
            persistent=False)
        fourier_dims = 2 + 2 * num_bands * 2
        self.projection = nn.Linear(fourier_dims, embed_dims)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, query, reference_points):
        uv = reference_points.float()
        scaled = uv.unsqueeze(-1) * self.frequencies.view(1, 1, 1, -1)
        encoded = torch.cat(
            [uv.unsqueeze(-1), scaled.sin(), scaled.cos()], dim=-1)
        encoded = encoded.flatten(start_dim=2)
        encoded = self.projection(encoded).to(query.dtype)
        return self.norm(query + encoded)


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
        self.spatial_norm = nn.LayerNorm(query_dims)
        self.ffn_norm = nn.LayerNorm(query_dims)
        self.linear1 = nn.Linear(query_dims, feedforward_channels)
        self.linear2 = nn.Linear(feedforward_channels, query_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, query_pos, spatial_features, reference_points,
                valid_ratios):
        spatial_value, spatial_shapes, spatial_level_start = spatial_features
        spatial = self.spatial_attention(
            self.spatial_norm(query), query_pos, spatial_value,
            spatial_shapes, spatial_level_start, reference_points,
            valid_ratios)
        query = query + self.dropout(spatial)

        ffn = self.linear2(self.dropout(F.gelu(
            self.linear1(self.ffn_norm(query)))))
        return query + self.dropout(ffn)


class GroundingDINO3DDecoder(nn.Module):

    def __init__(self, num_queries, query_dims=512, semantic_dims=256,
                 spatial_dims=512, num_layers=6, num_heads=8,
                 feedforward_channels=2048, num_feature_levels=4,
                 num_points=4, dropout=0.0):
        super().__init__()
        self.num_queries = num_queries
        self.semantic_query_projection = nn.Linear(semantic_dims, query_dims)
        self.query_embedding = nn.Embedding(num_queries, query_dims)
        self.init_norm = nn.LayerNorm(query_dims)
        self.query_position_embedding = FourierPositionEmbedding2D(query_dims)
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

    def initialize_query(self, semantic_query, reference_points):
        if semantic_query.ndim != 3:
            raise ValueError('semantic_query must have shape [N, Q, C]')
        if semantic_query.shape[1] != self.num_queries:
            raise ValueError(
                'semantic query count must match reconstruction query count')
        if reference_points.shape != semantic_query.shape[:2] + (2,):
            raise ValueError(
                'reference_points must have shape [N, Q, 2] matching query')
        learned_query = self.query_embedding.weight[None].expand(
            semantic_query.shape[0], -1, -1)
        projected_semantic = self.semantic_query_projection(semantic_query)
        query = self.init_norm(learned_query + projected_semantic)
        return self.query_position_embedding(query, reference_points)

    def forward(self, semantic_query, spatial_features, reference_points,
                valid_ratios):
        query = self.initialize_query(semantic_query, reference_points)
        query_pos = self.reference_projection(
            coordinate_to_encoding(reference_points))
        intermediate = []
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                spatial_features=spatial_features,
                reference_points=reference_points,
                valid_ratios=valid_ratios)
            intermediate.append(self.norm(query))
        return torch.stack(intermediate)
