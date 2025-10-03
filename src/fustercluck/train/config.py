"""Configuration dataclasses for text-stage training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class OptimizerConfig:
    lr: float = 2.5e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1e-8


@dataclass
class StageConfig:
    dataset_path: Path
    idx_path: Path
    tokenizer_path: Path
    max_steps: int = 2000
    seq_len: int = 2048
    micro_batch_size: int = 4
    gradient_accumulation: int = 8
    precision: str = "bf16"
    log_interval: int = 5
    eval_interval: int = 100
    checkpoint_dir: Path = Path("artifacts/checkpoints/stage0")
    model_dim: int = 1024
    model_layers: int = 4
    model_heads: int = 16
    model_kv_heads: int = 4
    mlp_ratio: float = 4.0
    rope_theta: int = 10000
    dropout: float = 0.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class CheckpointConfig:
    directory: Path
    keep_last: int = 3


@dataclass
class TrainerConfig:
    device: str = "mps"
    grad_clip: float = 1.0
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"
    precision: str = "bf16"
    dataloader_workers: int = 4  # Enable multi-threaded data loading for Apple Silicon
    pin_memory: bool = False  # MPS doesn't benefit from pin_memory
    persistent_workers: bool = True  # Keep workers alive between epochs
    prefetch_factor: int = 2  # Number of batches to prefetch per worker
    env: Optional[Dict[str, str]] = None
    checkpoint: Optional[CheckpointConfig] = None
    gradient_checkpointing: bool = False


__all__ = [
    "OptimizerConfig",
    "StageConfig",
    "CheckpointConfig",
    "TrainerConfig",
]
