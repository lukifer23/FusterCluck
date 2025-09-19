"""Gradient checkpointing helpers."""

from __future__ import annotations

import torch


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Enable PyTorch gradient checkpointing on all submodules supporting it."""

    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):  # HF modules
            module.gradient_checkpointing = True
        if hasattr(module, "forward"):
            module.forward = torch.utils.checkpoint.checkpoint_wrapper(module.forward, use_reentrant=False)  # type: ignore[assignment]
