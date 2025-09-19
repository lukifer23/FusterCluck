"""Utilities for streaming Hugging Face datasets and materializing tokenized corpora."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import sentencepiece as spm
from datasets import load_dataset  # type: ignore
from tqdm import tqdm


@dataclass(frozen=True)
class DatasetSpec:
    """Description of a Hugging Face dataset slice to stream."""

    name: str
    config: Optional[str]
    split: str
    text_field: str
    sample_limit: Optional[int] = None
    shuffle_buffer: Optional[int] = None

    @classmethod
    def parse(cls, raw: str) -> "DatasetSpec":
        """Parse specs of the form name:split:text_field[:sample_limit[:shuffle_buffer]]."""
        parts = raw.split(":")
        if len(parts) < 3:
            raise ValueError(
                "Dataset spec must be name:split:text_field[:sample_limit[:shuffle_buffer]]"
            )
        name, split, text_field = parts[:3]
        sample_limit = int(parts[3]) if len(parts) > 3 and parts[3] else None
        shuffle_buffer = int(parts[4]) if len(parts) > 4 and parts[4] else None
        return cls(
            name=name,
            config=None,
            split=split,
            text_field=text_field,
            sample_limit=sample_limit,
            shuffle_buffer=shuffle_buffer,
        )

    @classmethod
    def from_config(cls, raw: object) -> "DatasetSpec":
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str):
            return cls.parse(raw)
        if isinstance(raw, Mapping):
            data = dict(raw)
            if "name" not in data or "split" not in data or "text_field" not in data:
                raise ValueError(f"Dataset config requires name/split/text_field: {raw}")
            sample_limit = data.get("sample_limit")
            if sample_limit is not None:
                sample_limit = int(sample_limit)
            shuffle_buffer = data.get("shuffle_buffer")
            if shuffle_buffer is not None:
                shuffle_buffer = int(shuffle_buffer)
            return cls(
                name=str(data["name"]),
                config=data.get("config"),
                split=str(data["split"]),
                text_field=str(data["text_field"]),
                sample_limit=sample_limit,
                shuffle_buffer=shuffle_buffer,
            )
        raise TypeError(f"Unsupported dataset spec type: {type(raw)!r}")

    @property
    def key(self) -> str:
        parts = [self.name]
        if self.config:
            parts.append(self.config)
        parts.append(self.split)
        return ":".join(parts)


def extract_field(example: Dict[str, object], field: str) -> Optional[str]:
    value: object = example
    for key in field.split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def stream_dataset(spec: DatasetSpec, hf_token: Optional[str]) -> Iterator[str]:
    """Yield documents from a Hugging Face dataset according to `spec`."""
    args = [spec.name]
    if spec.config:
        args.append(spec.config)
    dataset = load_dataset(
        *args,
        split=spec.split,
        streaming=True,
        token=hf_token,
    )
    if spec.shuffle_buffer:
        dataset = dataset.shuffle(buffer_size=spec.shuffle_buffer, seed=42)
    produced = 0
    for row in dataset:  # type: ignore[misc]
        text = extract_field(row, spec.text_field)
        if not text:
            continue
        yield text
        produced += 1
        if spec.sample_limit and produced >= spec.sample_limit:
            break


def interleave_streams(
    specs: Sequence[DatasetSpec],
    hf_token: Optional[str],
) -> Iterator[Tuple[DatasetSpec, str]]:
    """Round-robin interleave the provided dataset specs."""
    if not specs:
        return
    active: List[DatasetSpec] = list(specs)
    generators = {spec: stream_dataset(spec, hf_token) for spec in specs}
    while active:
        spec = active.pop(0)
        gen = generators[spec]
        try:
            value = next(gen)
        except StopIteration:
            continue
        yield spec, value
        active.append(spec)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def write_idx_entry(handle, offset: int) -> None:
    handle.write(np.asarray([offset], dtype=np.int64).tobytes())


def materialize_tokenized_corpus(
    tokenizer_path: Path,
    specs: Sequence[DatasetSpec],
    output_prefix: Path,
    hf_token: Optional[str],
    *,
    target_tokens: Optional[int] = None,
    minimum_chars: int = 16,
    shuffle_buffer: int = 8192,
    seed: int = 13,
    overwrite: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Stream datasets, tokenize, and write `.bin/.idx` files.

    Returns statistics dictionary containing totals and per-dataset counts.
    Raises `RuntimeError` if the resulting corpus undershoots `target_tokens`.
    """

    if not specs:
        raise ValueError("At least one dataset spec is required")

    bin_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")
    if not overwrite and bin_path.exists() and idx_path.exists():
        raise FileExistsError(
            f"Tokenized corpus already exists: {bin_path} and {idx_path}. "
            "Set overwrite=True to rebuild."
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    processor = spm.SentencePieceProcessor()
    if not processor.Load(str(tokenizer_path)):
        raise RuntimeError(f"Failed to load SentencePiece model: {tokenizer_path}")
    eos_id = processor.eos_id()
    if eos_id < 0:
        raise RuntimeError("SentencePiece model is missing an EOS token")

    rng = random.Random(seed)
    per_dataset_sequences: Dict[str, int] = {spec.key: 0 for spec in specs}
    per_dataset_tokens: Dict[str, int] = {spec.key: 0 for spec in specs}

    offset_bytes = 0
    total_tokens = 0
    total_sequences = 0

    buffer: List[Tuple[str, str]] = []  # (dataset_key, text)
    buffer_limit = max(1, shuffle_buffer)
    iterator = interleave_streams(specs, hf_token)

    progress_total = target_tokens if target_tokens else None
    progress = tqdm(total=progress_total, unit="tok", unit_scale=True, dynamic_ncols=True)

    with bin_path.open("wb") as bin_handle, idx_path.open("wb") as idx_handle:
        write_idx_entry(idx_handle, 0)

        def flush_random() -> None:
            nonlocal offset_bytes, total_tokens, total_sequences
            if not buffer:
                return
            index = rng.randrange(len(buffer))
            dataset_key, raw_text = buffer.pop(index)
            cleaned = normalize_text(raw_text)
            if len(cleaned) < minimum_chars:
                return
            pieces = processor.EncodeAsIds(cleaned)
            if not pieces:
                return
            pieces.append(eos_id)
            arr = np.asarray(pieces, dtype=np.uint16)
            bin_handle.write(arr.tobytes())
            offset_bytes += arr.nbytes
            write_idx_entry(idx_handle, offset_bytes)
            total_tokens += arr.size
            total_sequences += 1
            per_dataset_sequences[dataset_key] += 1
            per_dataset_tokens[dataset_key] += int(arr.size)
            progress.update(arr.size)

        target_reached = False
        for spec, text in iterator:
            buffer.append((spec.key, text))
            if len(buffer) >= buffer_limit:
                flush_random()
                if target_tokens and total_tokens >= target_tokens:
                    target_reached = True
                    break
        if not target_reached:
            while buffer and (target_tokens is None or total_tokens < target_tokens):
                flush_random()
                if target_tokens and total_tokens >= target_tokens:
                    target_reached = True
                    break

    progress.close()

    if target_tokens and total_tokens < target_tokens:
        raise RuntimeError(
            f"Collected {total_tokens:,} tokens but target was {target_tokens:,}. "
            "Provide additional dataset slices or raise the limit."
        )

    stats = {
        "totals": {
            "tokens": total_tokens,
            "sequences": total_sequences,
            "bytes": offset_bytes,
        },
        "per_dataset_sequences": per_dataset_sequences,
        "per_dataset_tokens": per_dataset_tokens,
    }
    return stats


def token_count_from_idx(idx_path: Path) -> int:
    """Return the number of tokens implied by an `.idx` file."""
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    offsets = np.memmap(idx_path, mode="r", dtype=np.int64)
    if offsets.size < 1:
        raise ValueError(f"Index file {idx_path} is empty")
    last_offset = int(offsets[-1])
    return last_offset // 2
