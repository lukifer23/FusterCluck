"""Checkpointing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


class CheckpointManager:
    """Handles saving and pruning checkpoints."""

    def __init__(self, directory: Path, keep_last: int = 3) -> None:
        self.directory = directory
        self.keep_last = keep_last
        self.directory.mkdir(parents=True, exist_ok=True)

    def latest(self) -> Path | None:
        checkpoints = sorted(self.directory.glob("step-*.pt"), key=lambda p: p.stat().st_mtime)
        return checkpoints[-1] if checkpoints else None

    def save(self, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, **metadata: Any) -> Path:
        path = self.directory / f"step-{step:07d}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
        }, path)
        self._write_manifest(path, metadata)
        self._prune()
        return path

    def _write_manifest(self, path: Path, metadata: Dict[str, Any]) -> None:
        manifest = path.with_suffix(".json")
        with manifest.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _prune(self) -> None:
        checkpoints = sorted(self.directory.glob("step-*.pt"), key=lambda p: p.stat().st_mtime)
        excess = len(checkpoints) - self.keep_last
        for ckpt in checkpoints[:excess]:
            ckpt.unlink(missing_ok=True)
            manifest = ckpt.with_suffix(".json")
            manifest.unlink(missing_ok=True)
