"""Utilities for loading pre-tokenized corpora stored as binary files."""

from __future__ import annotations

import mmap
from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch


class TokenizedDataset:
    """Memory-mapped access to tokenized corpora stored in .bin/.idx format."""

    def __init__(self, data_path: Path, idx_path: Path) -> None:
        if not data_path.exists() or not idx_path.exists():
            raise FileNotFoundError("Tokenized dataset requires both .bin and .idx files")
        self.data_path = data_path
        self.idx_path = idx_path
        self._load_index()
        self._mmap_data()

    def _load_index(self) -> None:
        offsets = np.memmap(self.idx_path, mode="r", dtype=np.int64)
        if offsets.ndim != 1 or offsets.size < 2:
            raise ValueError("Index file must contain at least two offsets")
        self.offsets = offsets
        self.num_sequences = offsets.size - 1

    def _mmap_data(self) -> None:
        self.data_file = open(self.data_path, "rb")
        self.mmap = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(idx)
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        self.mmap.seek(start)
        byte_length = end - start
        buffer = self.mmap.read(byte_length)
        tokens = np.frombuffer(buffer, dtype=np.uint16)
        return torch.from_numpy(tokens.astype(np.int64))

    def close(self) -> None:
        self.mmap.close()
        self.data_file.close()


def pack_sequences(dataset: TokenizedDataset, max_length: int) -> Iterator[torch.Tensor]:
    """Yield packed sequences up to `max_length` tokens with cross-sample packing."""

    current: List[int] = []
    length = 0
    for idx in range(len(dataset)):
        tokens = dataset[idx].tolist()
        if length + len(tokens) >= max_length:
            remaining = max_length - length
            current.extend(tokens[:remaining])
            yield torch.tensor(current, dtype=torch.long)
            tokens = tokens[remaining:]
            current = []
            length = 0
            while len(tokens) >= max_length:
                yield torch.tensor(tokens[:max_length], dtype=torch.long)
                tokens = tokens[max_length:]
        current.extend(tokens)
        length = len(current)
    if current:
        yield torch.tensor(current, dtype=torch.long)
