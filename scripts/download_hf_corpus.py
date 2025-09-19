#!/usr/bin/env python3
"""Download large Hugging Face datasets into newline-delimited text files for Stage 1/2."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from datasets import load_dataset
from tqdm import tqdm


@dataclass
class DatasetSpec:
    name: str
    split: str
    text_field: str
    sample_limit: Optional[int] = None

    @classmethod
    def parse(cls, raw: str) -> "DatasetSpec":
        parts = raw.split(":")
        if len(parts) < 3:
            raise argparse.ArgumentTypeError(
                "Dataset spec must be in the form name:split:text_field[:max_samples]"
            )
        name, split, field = parts[:3]
        limit = int(parts[3]) if len(parts) > 3 else None
        return cls(name=name, split=split, text_field=field, sample_limit=limit)


def extract_field(example: dict, field: str) -> Optional[str]:
    value = example
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


def stream_dataset(spec: DatasetSpec, token: Optional[str]) -> Iterator[str]:
    dataset = load_dataset(spec.name, split=spec.split, streaming=True, token=token)
    counter = 0
    for row in dataset:
        text = extract_field(row, spec.text_field)
        if not text:
            continue
        yield text
        counter += 1
        if spec.sample_limit and counter >= spec.sample_limit:
            break


def download_corpora(specs: Iterable[DatasetSpec], output: Path, token: Optional[str]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as handle:
        for spec in specs:
            tqdm.write(f"Streaming {spec.name}:{spec.split} -> {output}")
            for text in tqdm(stream_dataset(spec, token), desc=spec.name, unit="samples"):
                handle.write(text.replace("\n", " ") + "\n")
                total += 1
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=DatasetSpec.parse,
        required=True,
        help="Dataset spec name:split:text_field[:max_samples]. "
        "Example: huggingface/RefinedWeb:train:text:500000",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv(args.token_env)
    if token is None:
        raise RuntimeError(
            f"Environment variable {args.token_env} is not set. "
            "Run `export HF_TOKEN=...` before invoking this script."
        )
    total = download_corpora(args.dataset, args.output, token)
    print(f"Wrote {total} samples to {args.output}")


if __name__ == "__main__":
    main()
