"""Text-aligned classification utilities shared by the 3D detection head."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def build_class_text_prototypes(text_tokens: Tensor, positive_map: dict,
                                num_classes: int) -> Tensor:
    """Aggregate GroundingDINO token features into zero-based class features."""
    if text_tokens.ndim != 2:
        raise ValueError('text_tokens must have shape [L, D]')
    prototypes = []
    for class_index in range(num_classes):
        token_indices = positive_map.get(class_index + 1)
        if not token_indices:
            raise ValueError(
                f'positive_map is missing class {class_index + 1}')
        indices = torch.as_tensor(token_indices, device=text_tokens.device,
                                  dtype=torch.long)
        if (indices < 0).any() or (indices >= text_tokens.shape[0]).any():
            raise ValueError(f'positive_map has invalid token index for class {class_index + 1}')
        prototype = text_tokens.index_select(0, indices).mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes, dim=0)


def build_multilabel_targets(num_queries: int, num_classes: int,
                             matched_query_indices: Tensor,
                             matched_labels: Tensor) -> Tensor:
    targets = torch.zeros(
        num_queries, num_classes, device=matched_query_indices.device,
        dtype=torch.float32)
    if matched_query_indices.numel() == 0:
        return targets
    if matched_query_indices.shape != matched_labels.shape:
        raise ValueError('matched query indices and labels must have equal shape')
    if (matched_labels < 0).any() or (matched_labels >= num_classes).any():
        raise ValueError('matched labels are outside the text class range')
    targets[matched_query_indices, matched_labels] = 1.0
    return targets


class TextAlignedClassificationHead(nn.Module):
    """Project 3D queries into the GroundingDINO text feature space."""

    def __init__(self, query_dim: int, text_dim: int, num_layers: int,
                 normalize: bool = True, bias_init: float = -4.6,
                 channel_first: bool = True) -> None:
        super().__init__()
        self.normalize = bool(normalize)
        self.text_dim = int(text_dim)
        self.channel_first = bool(channel_first)
        self.projections = nn.ModuleList(
            [nn.Linear(query_dim, text_dim) for _ in range(num_layers)])
        self.logit_scale = 1.0 / math.sqrt(text_dim)
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, features: list[Tensor], text_prototypes: Tensor,
                layer_ids: list[int] | None = None) -> list[Tensor]:
        if text_prototypes.ndim not in (2, 3):
            raise ValueError('text_prototypes must have shape [C, D] or [B, C, D]')
        if text_prototypes.shape[-1] != self.text_dim:
            raise ValueError('text prototype dimension does not match text head')
        if layer_ids is None:
            layer_ids = list(range(len(features)))
        outputs = []
        for feature, layer_id in zip(features, layer_ids):
            query = feature.transpose(1, 2) if self.channel_first else feature
            query = self.projections[layer_id](query)
            prototypes = text_prototypes.to(device=query.device, dtype=query.dtype)
            if self.normalize:
                query = F.normalize(query, dim=-1)
                prototypes = F.normalize(prototypes, dim=-1)
            logits = torch.matmul(query, prototypes.transpose(-1, -2))
            logits = logits * self.logit_scale + self.bias
            outputs.append(logits.transpose(1, 2) if self.channel_first else logits)
        return outputs
