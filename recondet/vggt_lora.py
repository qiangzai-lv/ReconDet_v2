from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LoRAInjectionSummary:
    replaced_modules: tuple[str, ...]
    trainable_parameters: int


class LoRALinear(nn.Module):
    """Low-rank residual adapter that preserves a frozen linear module."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float,
                 dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError('base_layer must be an nn.Linear')
        if rank <= 0:
            raise ValueError('rank must be positive')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in [0, 1)')

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        if 'bias_mask' in base_layer._buffers:
            self.register_buffer(
                'bias_mask', base_layer.bias_mask,
                persistent='bias_mask' not in
                base_layer._non_persistent_buffers_set)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        factory_kwargs = {
            'device': base_layer.weight.device,
            'dtype': base_layer.weight.dtype,
        }
        self.lora_a = nn.Linear(
            base_layer.in_features, self.rank, bias=False, **factory_kwargs)
        self.lora_b = nn.Linear(
            self.rank, base_layer.out_features, bias=False, **factory_kwargs)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs):
        bias = self.bias
        if bias is not None and hasattr(self, 'bias_mask'):
            bias = bias * self.bias_mask.to(bias.dtype)
        base = F.linear(inputs, self.weight, bias)
        update = self.lora_b(self.lora_a(self.dropout(inputs)))
        return base + update * self.scaling


_DEFAULT_CONFIG = {
    'enabled': False,
    'block_indices': (),
    'branches': ('frame_blocks', 'inter_frame_blocks'),
    'target_modules': ('attn.qkv', 'attn.proj'),
    'rank': 8,
    'alpha': 16,
    'dropout': 0.0,
    'gradient_checkpointing': False,
    'checkpoint_start_block': 0,
}


def _read_config(config: Mapping | None) -> dict:
    values = dict(_DEFAULT_CONFIG)
    if config is None:
        return values
    unknown = set(config) - set(values)
    if unknown:
        raise ValueError(f'Unknown vggt_lora_cfg keys: {sorted(unknown)}')
    values.update(dict(config))
    return values


def _validate_indices(indices: Sequence[int], depth: int,
                      name: str) -> tuple[int, ...]:
    if isinstance(indices, (str, bytes)):
        raise TypeError(f'{name} must be a sequence of integers')
    result = tuple(indices)
    if any(not isinstance(index, int) or isinstance(index, bool)
           for index in result):
        raise TypeError(f'{name} must contain integers')
    if len(set(result)) != len(result):
        raise ValueError(f'{name} contains duplicates')
    if any(index < 0 or index >= depth for index in result):
        raise ValueError(f'{name} contains an index out of range [0, {depth})')
    return result


def _resolve_module(root: nn.Module, path: str):
    current = root
    components = path.split('.')
    for component in components:
        if not hasattr(current, component):
            raise AttributeError(f'Target module {path!r} does not exist')
        current = getattr(current, component)
    return current


def _replace_module(root: nn.Module, path: str, module: nn.Module):
    components = path.split('.')
    parent = root
    for component in components[:-1]:
        parent = getattr(parent, component)
    setattr(parent, components[-1], module)


def configure_vggt_lora(
        aggregator: nn.Module,
        config: Mapping | None) -> LoRAInjectionSummary:
    """Freeze an aggregator and inject LoRA into configured attention layers."""
    values = _read_config(config)
    depth = int(getattr(aggregator, 'depth', 0))
    if depth <= 0:
        raise ValueError('VGGT aggregator depth must be positive')

    enabled = values['enabled']
    checkpoint_enabled = values['gradient_checkpointing']
    if not isinstance(enabled, bool):
        raise TypeError('enabled must be boolean')
    if not isinstance(checkpoint_enabled, bool):
        raise TypeError('gradient_checkpointing must be boolean')

    rank = values['rank']
    alpha = values['alpha']
    dropout = values['dropout']
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError('rank must be positive')
    if (not isinstance(alpha, (int, float)) or isinstance(alpha, bool)):
        raise TypeError('alpha must be numeric')
    if alpha <= 0:
        raise ValueError('alpha must be positive')
    if (not isinstance(dropout, (int, float)) or isinstance(dropout, bool)):
        raise TypeError('dropout must be numeric')
    if not 0.0 <= dropout < 1.0:
        raise ValueError('dropout must be in [0, 1)')

    checkpoint_start = values['checkpoint_start_block']
    if (not isinstance(checkpoint_start, int)
            or isinstance(checkpoint_start, bool)
            or checkpoint_start < 0 or checkpoint_start >= depth):
        raise ValueError('checkpoint_start_block is out of range')

    aggregator.requires_grad_(False)
    if not enabled:
        if hasattr(aggregator, 'configure_gradient_checkpointing'):
            aggregator.configure_gradient_checkpointing(False, 0)
        return LoRAInjectionSummary((), 0)

    block_indices = _validate_indices(
        values['block_indices'], depth, 'block_indices')
    if not block_indices:
        raise ValueError('block_indices must not be empty when LoRA is enabled')

    branches = tuple(values['branches'])
    targets = tuple(values['target_modules'])
    if not branches:
        raise ValueError('branches must not be empty')
    if not targets:
        raise ValueError('target_modules must not be empty')
    if len(set(branches)) != len(branches):
        raise ValueError('branches contains duplicates')
    if len(set(targets)) != len(targets):
        raise ValueError('target_modules contains duplicates')
    for branch in branches:
        if branch not in ('frame_blocks', 'inter_frame_blocks'):
            raise ValueError(f'Unknown VGGT LoRA branch: {branch}')
        if not hasattr(aggregator, branch):
            raise AttributeError(f'VGGT aggregator has no branch {branch!r}')

    replaced = []
    for branch in branches:
        blocks = getattr(aggregator, branch)
        for block_index in block_indices:
            block = blocks[block_index]
            for target in targets:
                base_layer = _resolve_module(block, target)
                if isinstance(base_layer, LoRALinear):
                    raise ValueError(
                        f'LoRA is already configured for '
                        f'{branch}.{block_index}.{target}')
                if not isinstance(base_layer, nn.Linear):
                    raise TypeError(
                        f'Target {branch}.{block_index}.{target} must be linear')
                _replace_module(
                    block, target,
                    LoRALinear(base_layer, rank, alpha, dropout))
                replaced.append(f'{branch}.{block_index}.{target}')

    if hasattr(aggregator, 'configure_gradient_checkpointing'):
        aggregator.configure_gradient_checkpointing(
            checkpoint_enabled, checkpoint_start)
    trainable = sum(parameter.numel() for parameter in aggregator.parameters()
                    if parameter.requires_grad)
    return LoRAInjectionSummary(tuple(replaced), trainable)


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return the complete set of adapter tensors in a module tree."""
    return {
        name: value
        for name, value in module.state_dict().items()
        if '.lora_a.' in name or '.lora_b.' in name
    }


def load_lora_state_dict(module: nn.Module,
                         state_dict: Mapping[str, torch.Tensor]) -> None:
    """Load adapters only after validating every key and tensor shape."""
    expected = lora_state_dict(module)
    expected_keys = set(expected)
    provided_keys = set(state_dict)
    missing = sorted(expected_keys - provided_keys)
    unexpected = sorted(provided_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(
            'Adapter keys do not match configured LoRA modules: '
            f'missing={missing}, unexpected={unexpected}')
    mismatched = [
        name for name in expected_keys
        if tuple(state_dict[name].shape) != tuple(expected[name].shape)
    ]
    if mismatched:
        raise RuntimeError(
            f'Adapter tensor shapes do not match: {sorted(mismatched)}')
    module.load_state_dict(dict(state_dict), strict=False)


def enable_lora_parameters(module: nn.Module) -> None:
    """Freeze a module tree and re-enable only its LoRA parameters."""
    module.requires_grad_(False)
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.lora_a.requires_grad_(True)
            child.lora_b.requires_grad_(True)


@contextmanager
def vggt_feature_grad_context(enabled: bool):
    """Enable VGGT graphs only when adapters and the outer graph are active."""
    if enabled and torch.is_grad_enabled():
        yield
    else:
        with torch.no_grad():
            yield
