#!/usr/bin/env python3
"""Convert raw text files to packed uint16 token files (.bin/.idx) with streaming writes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def iter_corpus(paths: Iterable[Path]) -> Iterator[str]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                yield line.strip()


def encode_corpus(
    processor: spm.SentencePieceProcessor,
    text_iter: Iterable[str],
    output_prefix: Path,
    *,
    minimum_chars: int = 1,
) -> tuple[int, int]:
    bin_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    eos_id = processor.eos_id()
    if eos_id < 0:
        raise RuntimeError("SentencePiece model is missing an EOS token")

    total_tokens = 0
    total_sequences = 0
    offset_bytes = 0

    with bin_path.open("wb") as bin_handle, idx_path.open("wb") as idx_handle:
        idx_handle.write(np.asarray([0], dtype=np.int64).tobytes())
        for line in tqdm(text_iter, desc="encoding", unit="lines"):
            if len(line) < minimum_chars:
                continue
            pieces = processor.EncodeAsIds(line)
            if not pieces:
                continue
            pieces.append(eos_id)
            arr = np.asarray(pieces, dtype=np.uint16)
            bin_handle.write(arr.tobytes())
            offset_bytes += arr.nbytes
            idx_handle.write(np.asarray([offset_bytes], dtype=np.int64).tobytes())
            total_tokens += arr.size
            total_sequences += 1

    return total_tokens, total_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path, help="SentencePiece model path")
    parser.add_argument("output", type=Path, help="Output prefix for .bin/.idx")
    parser.add_argument("corpus", nargs="+", type=Path, help="UTF-8 text files")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=1,
        help="Discard lines shorter than this many characters",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    processor = spm.SentencePieceProcessor()
    if not processor.Load(str(args.tokenizer)):
        raise RuntimeError(f"Failed to load SentencePiece model: {args.tokenizer}")
    tokens, sequences = encode_corpus(
        processor,
        iter_corpus(args.corpus),
        args.output,
        minimum_chars=args.min_chars,
    )
    LOGGER.info(
        "Wrote %s and %s (%d tokens across %d sequences)",
        args.output.with_suffix(".bin"),
        args.output.with_suffix(".idx"),
        tokens,
        sequences,
    )


if __name__ == "__main__":
    main()
