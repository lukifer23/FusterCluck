#!/usr/bin/env python3
"""Convert raw text files to packed uint16 token files (.bin/.idx)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import sentencepiece as spm

LOGGER = logging.getLogger(__name__)


def encode_corpus(processor: spm.SentencePieceProcessor, text_paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    tokens: list[np.ndarray] = []
    total = 0
    for path in text_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                pieces = processor.EncodeAsIds(line)
                arr = np.asarray(pieces + [processor.eos_id()], dtype=np.uint16)
                tokens.append(arr)
                total += arr.size
                offsets.append(total * 2)  # uint16 -> 2 bytes per token
    flat = np.concatenate(tokens)
    return flat.view(np.uint16), np.asarray(offsets, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path, help="SentencePiece model path")
    parser.add_argument("output", type=Path, help="Output prefix for .bin/.idx")
    parser.add_argument("corpus", nargs="+", type=Path, help="UTF-8 text files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    processor = spm.SentencePieceProcessor()
    processor.Load(str(args.tokenizer))
    tokens, offsets = encode_corpus(processor, args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bin_path = args.output.with_suffix(".bin")
    idx_path = args.output.with_suffix(".idx")
    tokens.tofile(bin_path)
    offsets.tofile(idx_path)
    num_tokens = tokens.size
    num_sequences = offsets.size - 1
    LOGGER.info(
        "Wrote %s and %s (%d tokens across %d sequences)",
        bin_path,
        idx_path,
        num_tokens,
        num_sequences,
    )


if __name__ == "__main__":
    main()
