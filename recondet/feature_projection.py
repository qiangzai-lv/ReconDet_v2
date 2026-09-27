import copy

import torch
import torch.nn as nn

from vggt_omega.models.heads.dense_head import DenseHead


class VGGTFeatureProjector(nn.Module):
    def __init__(self, dense_head, dim_in, out_channels):
        super().__init__()
        self.patch_size = dense_head.patch_size
        self.intermediate_layer_idx = dense_head.intermediate_layer_idx
        self.norm = copy.deepcopy(dense_head.norm)
        self.output_projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=oc,
                kernel_size=1,
                stride=1,
                padding=0)
            for oc in out_channels
        ])

    def forward(self, aggregated_tokens_list, images, patch_token_start):
        batch_size, num_frames, _, height, width = images.shape
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        feature_maps = []

        for feature_idx, layer_idx in enumerate(
                self.intermediate_layer_idx):
            feature = aggregated_tokens_list[layer_idx]
            feature = feature[:, :, patch_token_start:]
            if feature.dtype != torch.float32:
                feature = feature.float()
            feature = feature.reshape(
                batch_size * num_frames, -1, feature.shape[-1])
            feature = self.norm(feature)
            feature = feature.permute(0, 2, 1).reshape(
                batch_size * num_frames, feature.shape[-1],
                patch_height, patch_width)
            feature = DenseHead._apply_pos_embed(
                self, feature, width, height)
            feature = self.output_projects[feature_idx](feature)
            channels = feature.shape[1]
            feature_maps.append(
                feature.reshape(
                    batch_size, num_frames, channels,
                    feature.shape[-2], feature.shape[-1]).contiguous())

        return feature_maps


class VGGTDetectionPyramid(nn.Module):
    """Convert cached VGGT tokens into GroundingDINO feature levels."""

    def __init__(self, dense_head, out_channels=256, norm_groups=32):
        super().__init__()
        if out_channels <= 0:
            raise ValueError('out_channels must be positive')
        if norm_groups <= 0 or out_channels % norm_groups != 0:
            raise ValueError('norm_groups must divide out_channels')
        self.patch_size = dense_head.patch_size
        self.intermediate_layer_idx = tuple(
            dense_head.intermediate_layer_idx)
        self.norm = copy.deepcopy(dense_head.norm)
        self.projects = copy.deepcopy(dense_head.projects)
        self.resize_layers = copy.deepcopy(dense_head.resize_layers)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    project.out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1),
                nn.GroupNorm(norm_groups, out_channels))
            for project in self.projects
        ])

    def forward(self, aggregated_tokens_list, images, patch_token_start):
        batch_size, num_frames, _, height, width = images.shape
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        feature_maps = []

        for feature_idx, layer_idx in enumerate(
                self.intermediate_layer_idx):
            feature = aggregated_tokens_list[layer_idx]
            if feature is None:
                raise ValueError(
                    f'VGGT aggregator did not cache layer {layer_idx}')
            feature = feature[:, :, patch_token_start:]
            if feature.dtype != torch.float32:
                feature = feature.float()
            feature = feature.reshape(
                batch_size * num_frames, -1, feature.shape[-1])
            feature = self.norm(feature)
            feature = feature.permute(0, 2, 1).reshape(
                batch_size * num_frames, feature.shape[-1],
                patch_height, patch_width)
            feature = self.projects[feature_idx](feature)
            feature = DenseHead._apply_pos_embed(
                self, feature, width, height)
            feature = self.resize_layers[feature_idx](feature)
            feature = self.adapters[feature_idx](feature)
            feature_maps.append(feature.reshape(
                batch_size, num_frames, feature.shape[1],
                feature.shape[2], feature.shape[3]).contiguous())

        return feature_maps
