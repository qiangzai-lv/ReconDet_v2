import torch
import torch.nn as nn

from vggt_omega.models.heads.dense_head import DenseHead


class VGGTFeatureProjector(nn.Module):
    def __init__(self, dense_head, dim_in, out_channels):
        super().__init__()
        self.patch_size = dense_head.patch_size
        self.intermediate_layer_idx = dense_head.intermediate_layer_idx
        self.norm = dense_head.norm
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
