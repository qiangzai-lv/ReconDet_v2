"""Small, dependency-free LoRA adapters for VGGT-Omega attention qkv layers."""

from __future__ import annotations

import torch
from torch import nn


LORA_SCOPES = {
    "patch_embed", "frame", "frame_inter_frame", "frame/inter-frame",
    "all_aggregator",
}


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable low-rank residual.

    The wrapped layer is kept intact so Omega checkpoints remain loadable and
    masked qkv bias behavior is preserved. LoRA B is zero initialized, making
    the adapter an exact identity before training.
    """

    def __init__(self, base_layer: nn.Module, rank: int, alpha: float,
                 dropout: float, renorm: bool = True) -> None:
        super().__init__()
        if not hasattr(base_layer, "in_features") or not hasattr(base_layer, "out_features"):
            raise TypeError("LoRA target must expose in_features/out_features")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.renorm = bool(renorm)
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.merged = False

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        if self.merged:
            return base
        update = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        if not self.renorm or base.shape[-1] % 3:
            return base + update

        # Omega qkv is concatenated as [Q, K, V]. Preserve each component's
        # pretrained norm while allowing its direction to adapt.
        width = base.shape[-1] // 3
        base_parts = base.split(width, dim=-1)
        combined_parts = (base + update).split(width, dim=-1)
        normalized = []
        for original, combined in zip(base_parts, combined_parts):
            original_norm = original.float().norm(dim=-1, keepdim=True)
            combined_norm = combined.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            normalized.append((combined.float() * (original_norm / combined_norm)).to(combined.dtype))
        return torch.cat(normalized, dim=-1)

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        delta = self.lora_B.weight @ self.lora_A.weight
        self.base_layer.weight.add_(delta.to(self.base_layer.weight.dtype) * self.scaling)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        delta = self.lora_B.weight @ self.lora_A.weight
        self.base_layer.weight.sub_(delta.to(self.base_layer.weight.dtype) * self.scaling)
        self.merged = False


def _target_names(aggregator: nn.Module, scope: str,
                  layer_indices: list[int] | tuple[int, ...] | None) -> list[str]:
    if scope not in LORA_SCOPES:
        raise ValueError(f"Unknown vggt_lora_scope={scope!r}; expected one of {sorted(LORA_SCOPES)}")
    depth = int(aggregator.depth)
    if layer_indices is None:
        selected_layers = list(range(depth))
    else:
        selected_layers = sorted(set(int(index) for index in layer_indices))
        if not selected_layers:
            raise ValueError('vggt_lora_layer_indices must not be empty')
        invalid = [index for index in selected_layers
                   if index < 0 or index >= depth]
        if invalid:
            raise ValueError(
                f'vggt_lora_layer_indices outside [0, {depth - 1}]: {invalid}')
    names: list[str] = []
    if scope in {"patch_embed", "all_aggregator"}:
        names.extend(
            f"patch_embed.blocks.{i}.attn.qkv" for i in selected_layers)
    if scope in {"frame", "frame_inter_frame", "frame/inter-frame", "all_aggregator"}:
        names.extend(f"frame_blocks.{i}.attn.qkv" for i in selected_layers)
        names.extend(
            f"inter_frame_blocks.{i}.attn.qkv" for i in selected_layers)
    return names


def inject_vggt_lora(aggregator: nn.Module, scope: str, rank: int = 8,
                     alpha: float = 8.0, dropout: float = 0.1,
                     renorm: bool = True,
                     layer_indices: list[int] | tuple[int, ...] | None = None
                     ) -> list[str]:
    """Replace selected Omega qkv linears and return injected module names."""
    injected = []
    for name in _target_names(aggregator, scope, layer_indices):
        parent_name, child_name = name.rsplit(".", 1)
        parent = aggregator.get_submodule(parent_name)
        current = getattr(parent, child_name)
        if isinstance(current, LoRALinear):
            injected.append(name)
            continue
        setattr(parent, child_name, LoRALinear(current, rank, alpha, dropout, renorm))
        injected.append(name)
    return injected


def configure_vggt_lora(aggregator: nn.Module, enabled: bool) -> None:
    """Freeze Omega base parameters and optionally enable adapter gradients."""
    for name, parameter in aggregator.named_parameters():
        is_adapter = ".lora_A." in name or ".lora_B." in name
        parameter.requires_grad = bool(enabled and is_adapter)
    # The expression above intentionally only enables adapter tensors. Keep
    # this explicit check to make failures obvious if naming changes.
    if enabled and not any(p.requires_grad for p in aggregator.parameters()):
        raise RuntimeError("LoRA was enabled but no adapter parameters were found")


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()
            if ".lora_A." in name or ".lora_B." in name}


def load_lora_state_dict(module: nn.Module, state: dict[str, torch.Tensor], strict: bool = True):
    current = lora_state_dict(module)
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    if strict and (missing or unexpected):
        raise RuntimeError(f"LoRA checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    for name, value in state.items():
        if name in current:
            target = module.state_dict()[name]
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    return missing, unexpected
