"""Shared utility helpers."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator

import torch


def get_device(device_preference: str = "mps") -> torch.device:
    if device_preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_preference.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_preference)
    return torch.device("cpu")


def amp_autocast(precision: str, device: torch.device) -> contextlib.AbstractContextManager:
    if precision == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def ensure_dir(path) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
