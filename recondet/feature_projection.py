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
    """Build shared VGGT feature levels for 2D and 3D detection."""

    def __init__(self, dense_head, out_channels=256,
                 geometry_out_channels=None, norm_groups=32):
        super().__init__()
        if out_channels <= 0:
            raise ValueError('out_channels must be positive')
        if norm_groups <= 0 or out_channels % norm_groups != 0:
            raise ValueError('norm_groups must divide out_channels')
        if (geometry_out_channels is not None
                and geometry_out_channels <= 0):
            raise ValueError('geometry_out_channels must be positive')
        if (geometry_out_channels is not None
                and geometry_out_channels % norm_groups != 0):
            raise ValueError(
                'norm_groups must divide geometry_out_channels')
        self.patch_size = dense_head.patch_size
        self.intermediate_layer_idx = tuple(
            dense_head.intermediate_layer_idx)
        self.out_channels = out_channels
        self.geometry_out_channels = geometry_out_channels
        shared_channels = geometry_out_channels or out_channels
        self.norm = copy.deepcopy(dense_head.norm)
        self.projects = copy.deepcopy(dense_head.projects)
        self.resize_layers = copy.deepcopy(dense_head.resize_layers)
        self.adapters = nn.ModuleList([
            nn.Conv2d(
                project.out_channels,
                shared_channels,
                kernel_size=3,
                stride=2,
                padding=1)
            for project in self.projects
        ])
        self.detection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(shared_channels, out_channels, kernel_size=1),
                nn.GroupNorm(norm_groups, out_channels))
            for _ in self.projects
        ])
        self.geometry_norms = nn.ModuleList()
        if geometry_out_channels is not None:
            self.geometry_norms.extend([
                nn.GroupNorm(norm_groups, geometry_out_channels)
                for _ in self.projects
            ])
        self._initialize_detection_selectors()

    def _initialize_detection_selectors(self):
        """Make the 2D head select the legacy adapter channels."""
        with torch.no_grad():
            for head in self.detection_heads:
                projection = head[0]
                projection.weight.zero_()
                projection.bias.zero_()
                selected = min(
                    projection.in_channels, projection.out_channels)
                channel_index = torch.arange(selected)
                projection.weight[
                    channel_index, channel_index, 0, 0] = 1

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Migrate legacy ``Conv + GroupNorm`` 2D adapter weights."""
        for index, adapter in enumerate(self.adapters):
            old_conv_weight_key = f'{prefix}adapters.{index}.0.weight'
            old_conv_bias_key = f'{prefix}adapters.{index}.0.bias'
            old_norm_weight_key = f'{prefix}adapters.{index}.1.weight'
            old_norm_bias_key = f'{prefix}adapters.{index}.1.bias'
            new_conv_weight_key = f'{prefix}adapters.{index}.weight'
            new_conv_bias_key = f'{prefix}adapters.{index}.bias'
            detection_prefix = f'{prefix}detection_heads.{index}'

            if (old_conv_weight_key in state_dict
                    and new_conv_weight_key not in state_dict):
                old_weight = state_dict.pop(old_conv_weight_key)
                migrated_weight = adapter.weight.detach().clone()
                copied_channels = min(
                    old_weight.shape[0], migrated_weight.shape[0])
                migrated_weight[:copied_channels].copy_(
                    old_weight[:copied_channels])
                state_dict[new_conv_weight_key] = migrated_weight
            if (old_conv_bias_key in state_dict
                    and new_conv_bias_key not in state_dict):
                old_bias = state_dict.pop(old_conv_bias_key)
                migrated_bias = adapter.bias.detach().clone()
                copied_channels = min(
                    old_bias.shape[0], migrated_bias.shape[0])
                migrated_bias[:copied_channels].copy_(
                    old_bias[:copied_channels])
                state_dict[new_conv_bias_key] = migrated_bias

            norm_weight_key = f'{detection_prefix}.1.weight'
            norm_bias_key = f'{detection_prefix}.1.bias'
            if old_norm_weight_key in state_dict:
                state_dict[norm_weight_key] = state_dict.pop(
                    old_norm_weight_key)
            if old_norm_bias_key in state_dict:
                state_dict[norm_bias_key] = state_dict.pop(
                    old_norm_bias_key)

            projection = self.detection_heads[index][0]
            projection_weight_key = f'{detection_prefix}.0.weight'
            projection_bias_key = f'{detection_prefix}.0.bias'
            state_dict.setdefault(
                projection_weight_key,
                projection.weight.detach().clone())
            state_dict.setdefault(
                projection_bias_key,
                projection.bias.detach().clone())

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def forward(self, aggregated_tokens_list, images, patch_token_start,
                return_geometry=False):
        if return_geometry and self.geometry_out_channels is None:
            raise RuntimeError(
                'geometry_out_channels is required for 3D features')
        batch_size, num_frames, _, height, width = images.shape
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        detection_feature_maps = []
        geometry_feature_maps = []

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
            shared_feature = self.adapters[feature_idx](feature)
            detection_feature = self.detection_heads[feature_idx](
                shared_feature)
            detection_feature_maps.append(detection_feature.reshape(
                batch_size, num_frames, detection_feature.shape[1],
                detection_feature.shape[2],
                detection_feature.shape[3]).contiguous())
            if return_geometry:
                geometry_feature = self.geometry_norms[feature_idx](
                    shared_feature)
                geometry_feature_maps.append(geometry_feature.reshape(
                    batch_size, num_frames, geometry_feature.shape[1],
                    geometry_feature.shape[2],
                    geometry_feature.shape[3]).contiguous())

        if return_geometry:
            return detection_feature_maps, geometry_feature_maps
        return detection_feature_maps
