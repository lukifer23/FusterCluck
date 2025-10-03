"""Utilities for loading pre-tokenized corpora stored as binary files."""

from __future__ import annotations

import mmap
from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch


class TokenizedDataset:
    """Memory-mapped access to tokenized corpora stored in .bin/.idx format."""

    def __init__(self, data_path: Path, idx_path: Path, preload: bool = False) -> None:
        if not data_path.exists() or not idx_path.exists():
            raise FileNotFoundError("Tokenized dataset requires both .bin and .idx files")
        self.data_path = data_path
        self.idx_path = idx_path
        self.preload = preload
        self._load_index()
        if preload:
            self._preload_data()
        else:
            self._mmap_data()

    def _load_index(self) -> None:
        offsets = np.memmap(self.idx_path, mode="r", dtype=np.int64)
        if offsets.ndim != 1 or offsets.size < 2:
            raise ValueError("Index file must contain at least two offsets")
        self.offsets = offsets
        self.num_sequences = offsets.size - 1

    def _mmap_data(self) -> None:
        # For multiprocessing compatibility, we'll open the file on-demand
        # rather than keeping mmap objects that can't be pickled
        pass

    def _preload_data(self) -> None:
        """Preload all data into memory for faster access."""
        # Read all data at once
        with open(self.data_path, "rb") as f:
            all_data = f.read()

        # Convert to numpy array of uint16 tokens
        tokens = np.frombuffer(all_data, dtype=np.uint16)

        # Pre-allocate list for sequences
        self.sequences = []

        # Split into individual sequences based on offsets
        for i in range(self.num_sequences):
            start = int(self.offsets[i])
            end = int(self.offsets[i + 1])
            # Convert from byte offsets to token offsets (2 bytes per uint16)
            token_start = start // 2
            token_end = end // 2
            sequence = torch.from_numpy(tokens[token_start:token_end].astype(np.int64))
            self.sequences.append(sequence)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(idx)

        if self.preload:
            return self.sequences[idx]
        else:
            # Open file on-demand for multiprocessing compatibility
            start = int(self.offsets[idx])
            end = int(self.offsets[idx + 1])
            with open(self.data_path, "rb") as f:
                f.seek(start)
                buffer = f.read(end - start)
            tokens = np.frombuffer(buffer, dtype=np.uint16)
            return torch.from_numpy(tokens.astype(np.int64))

    def close(self) -> None:
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'data_file'):
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
