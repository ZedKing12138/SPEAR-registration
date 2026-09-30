"""Explicit CPU/CUDA selection shared by training and inference.

CUDA is the main command-line workflow. CPU remains available for development
and small tests. A requested CUDA device is never silently replaced by CPU.
All numerical model operations currently use float32: mixed precision is not
enabled implicitly because dense sampling and local NCC need separate accuracy
validation before an AMP policy can safely be chosen.
"""

from __future__ import annotations

import torch


def resolve_device(value: str | torch.device) -> str:
    """Validate a device and return an explicit canonical device string.

    CUDA indices refer to the devices visible to this process, so setting
    ``CUDA_VISIBLE_DEVICES=2`` and selecting ``cuda:0`` uses physical GPU 2.
    Supplying ``cuda`` selects PyTorch's current CUDA device, without changing
    the global current-device setting. Explicit indices are also passed to
    synchronization and memory-accounting calls elsewhere in the engine.
    """
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'") from error
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("CPU selection must be 'cpu', without a device index")
        return "cpu"
    if device.type != "cuda":
        raise ValueError("Only CPU and CUDA execution are supported; choose cpu or cuda:<index>")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested ({value}) but is unavailable. Install a CUDA-enabled "
            "PyTorch build and check the NVIDIA driver/CUDA_VISIBLE_DEVICES; "
            "use --device cpu explicitly for CPU execution. No CPU fallback was applied."
        )
    index = torch.cuda.current_device() if device.index is None else device.index
    count = torch.cuda.device_count()
    if index >= count:
        raise ValueError(
            f"CUDA device index {index} is out of range: {count} visible GPU(s). "
            "Indices are relative to CUDA_VISIBLE_DEVICES."
        )
    return f"cuda:{index}"
