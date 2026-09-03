from contextlib import nullcontext

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def is_npu_available() -> bool:
    return torch_npu is not None and torch.npu.is_available()


def get_device() -> torch.device:
    if is_npu_available():
        return torch.device('npu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == 'npu':
        return torch.float16
    if device.type == 'cuda':
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def autocast(device: torch.device, enabled: bool = True):
    if not enabled or device.type == 'cpu':
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=get_amp_dtype(device))
