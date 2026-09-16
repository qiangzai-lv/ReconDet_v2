import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import MultiScaleDeformableAttention


def project_queries_to_views(query_xyz, extrinsics, intrinsics, image_shape):
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    camera_points = torch.einsum(
        'bvij,bqj->bvqi', rotation, query_xyz) + translation[:, :, None]
    depth = camera_points[..., 2]
    pixels = torch.einsum('bvij,bvqj->bvqi', intrinsics, camera_points)
    pixels = pixels[..., :2] / depth[..., None].clamp_min(1e-5)

    height, width = image_shape
    reference_points = pixels / pixels.new_tensor([width, height])
    visible = depth > 1e-5
    visible &= torch.isfinite(reference_points).all(dim=-1)
    visible &= (reference_points[..., 0] >= 0)
    visible &= (reference_points[..., 0] <= 1)
    visible &= (reference_points[..., 1] >= 0)
    visible &= (reference_points[..., 1] <= 1)
    return reference_points.permute(0, 2, 1, 3), visible.permute(0, 2, 1)


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
                view_mask):
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
        reference_per_view = reference_points.permute(0, 2, 1, 3).reshape(
            batch_size * num_views, num_queries, 1, 2)
        reference_per_view = reference_per_view.nan_to_num(0.5).clamp(0., 1.)
        reference_per_view = reference_per_view.expand(
            -1, -1, self.num_feature_levels, -1)

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
        view_scores = view_scores.masked_fill(~valid_views, -torch.inf)
        view_weights = torch.nan_to_num(view_scores.softmax(dim=1), nan=0.)
        return (view_weights[..., None] * attended).sum(dim=1)


class GeometryAwareDecoderLayer(nn.Module):

    def __init__(self, embed_dims, num_heads, feedforward_channels,
                 num_feature_levels, num_points, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = ProjectedDeformableCrossAttention(
            embed_dims, num_heads, num_feature_levels, num_points, dropout)
        self.linear1 = nn.Linear(embed_dims, feedforward_channels)
        self.linear2 = nn.Linear(feedforward_channels, embed_dims)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, query_pos, feature_maps, reference_points,
                view_mask):
        query_norm = self.norm1(query)
        self_attended = self.self_attn(
            query_norm + query_pos,
            query_norm + query_pos,
            query_norm,
            need_weights=False)[0]
        query = query + self.dropout(self_attended)

        cross_attended = self.cross_attn(
            self.norm2(query), query_pos, feature_maps, reference_points,
            view_mask)
        query = query + self.dropout(cross_attended)

        ffn = self.linear2(self.dropout(F.gelu(self.linear1(self.norm3(query)))))
        return query + self.dropout(ffn)


class GeometryAwareDeformableDecoder(nn.Module):

    def __init__(self, embed_dims, num_layers, num_heads,
                 feedforward_channels, num_feature_levels, num_points=4,
                 dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            GeometryAwareDecoderLayer(
                embed_dims, num_heads, feedforward_channels,
                num_feature_levels, num_points, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, query, feature_maps, query_xyz, extrinsics, intrinsics,
                image_shape,
                position_embedding, query_projection, center_branches):
        if len(center_branches) != len(self.layers):
            raise ValueError(
                'Each decoder layer requires one center regression branch')

        intermediate = []
        intermediate_references = []
        for layer_id, layer in enumerate(self.layers):
            query_pos = query_projection(
                position_embedding(query_xyz, input_range=None)).transpose(1, 2)
            image_references, view_mask = project_queries_to_views(
                query_xyz, extrinsics, intrinsics, image_shape)
            query = layer(
                query, query_pos, feature_maps, image_references, view_mask)
            output = self.norm(query)

            center_delta = center_branches[layer_id](
                output.transpose(1, 2)).transpose(1, 2)
            refined_query_xyz = query_xyz + center_delta

            intermediate.append(output.transpose(1, 2))
            intermediate_references.append(refined_query_xyz)
            query_xyz = refined_query_xyz.detach()

        return intermediate, intermediate_references
