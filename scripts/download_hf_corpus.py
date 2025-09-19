#!/usr/bin/env python3
"""Download large Hugging Face datasets into newline-delimited text files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from fustercluck.data.corpus_builder import (
    DatasetSpec,
    normalize_text,
    stream_dataset,
)


def download_corpora(
    specs: Iterable[DatasetSpec],
    output: Path,
    token: str | None,
    minimum_chars: int = 0,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as handle:
        for spec in specs:
            desc = f"{spec.name}:{spec.split}"
            iterator = stream_dataset(spec, token)
            for text in tqdm(iterator, desc=desc, unit="samples"):
                cleaned = normalize_text(text)
                if len(cleaned) < minimum_chars:
                    continue
                handle.write(cleaned + "\n")
                total += 1
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec (name:split:text_field[:sample_limit[:shuffle_buffer]]).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output text file (newline-delimited).",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing your Hugging Face token (default: HF_TOKEN)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=0,
        help="Discard samples shorter than this many characters (after normalization)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv(args.token_env)
    specs = [DatasetSpec.from_config(spec) for spec in args.dataset]
    total = download_corpora(specs, args.output, token, args.min_chars)
    print(f"Wrote {total} samples to {args.output}")


if __name__ == "__main__":
    main()
