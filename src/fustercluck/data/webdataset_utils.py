"""Helper utilities for WebDataset ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import webdataset as wds


@dataclass
class WebDatasetConfig:
    urls: Sequence[str]
    batch_size: int
    shuffle_buffer: int = 2048
    decode: str = "pil"
    handler: str = "warn"


def build_webdataset(cfg: WebDatasetConfig) -> wds.WebDataset:
    if not cfg.urls:
        raise ValueError("No WebDataset shards provided")
    dataset = wds.WebDataset(cfg.urls, handler=cfg.handler)
    dataset = dataset.shuffle(cfg.shuffle_buffer)
    dataset = dataset.decode(cfg.decode)
    dataset = dataset.batched(cfg.batch_size)
    return dataset


def list_tar_shards(root: Path, pattern: str = "*.tar") -> List[str]:
    return sorted(str(path) for path in root.glob(pattern))
