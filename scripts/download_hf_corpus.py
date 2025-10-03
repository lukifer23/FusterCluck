#!/usr/bin/env python3
"""Download large-scale Hugging Face corpora for FusterCluck pre-training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from fustercluck.data.corpus_builder import (
    DatasetSpec,
    normalize_text,
    stream_dataset,
)


def download_domain_corpus(
    domain: str,
    output_dir: Path,
    token: str,
    target_samples: int,
    min_chars: int = 200
) -> Dict[str, int]:
    """Download corpus for a specific domain."""

    # Domain-specific dataset configurations
    # Note: Using HuggingFaceFW datasets as alternatives since academic/code datasets have access issues
    domain_datasets = {
        "physics": [
            "HuggingFaceFW/fineweb-edu:sample-10BT:train:text::15000",  # Educational content (physics/math heavy)
        ],
        "math": [
            "HuggingFaceFW/fineweb-edu:sample-10BT:train:text::15000",  # Educational content (physics/math heavy)
        ],
        "code": [
            "HuggingFaceFW/fineweb-edu:sample-10BT:train:text::20000",  # Educational content (programming tutorials)
            "HuggingFaceFW/fineweb:sample-10BT:train:text::15000",  # General web content (code examples)
        ],
        "general": [
            "HuggingFaceFW/fineweb:sample-10BT:train:text::5000",  # General web content
        ],
        "chess": [
            # Chess data is already downloaded via other scripts
            "HuggingFaceFW/fineweb-edu:sample-10BT:train:text::1000",  # Educational content (chess tutorials)
        ]
    }

    if domain not in domain_datasets:
        raise ValueError(f"Unknown domain: {domain}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "02d"

    total_samples = 0
    domain_stats = {"domain": domain, "datasets": [], "total_samples": 0}

    for dataset_spec in domain_datasets[domain]:
        print(f"📥 Downloading {dataset_spec} for {domain}")

        try:
            # Parse dataset spec
            parts = dataset_spec.split(":")
            if len(parts) < 3:
                continue

            # Handle specs like "name:config:split:text::sample_limit"
            sample_limit = None
            shuffle_buffer = None
            if len(parts) > 4 and parts[4]:
                sample_limit = int(parts[4])
            elif len(parts) > 5 and parts[5]:
                sample_limit = int(parts[5])
            if len(parts) > 6 and parts[6]:
                shuffle_buffer = int(parts[6])

            spec = DatasetSpec(
                name=parts[0],
                config=parts[1] if parts[1] != "default" else None,
                split=parts[2],
                text_field=parts[3],
                sample_limit=sample_limit,
                shuffle_buffer=shuffle_buffer,
            )

            # Download and append to file
            dataset_samples = 0
            iterator = stream_dataset(spec, token)

            with output_file.open("a", encoding="utf-8") as handle:
                for text in tqdm(iterator, desc=f"{spec.name}:{spec.split}", unit="samples"):
                    cleaned = normalize_text(text)
                    if len(cleaned) < min_chars:
                        continue

                    handle.write(cleaned + "\n")
                    dataset_samples += 1

                    if spec.sample_limit and dataset_samples >= spec.sample_limit:
                        break

            domain_stats["datasets"].append({
                "dataset": spec.name,
                "samples": dataset_samples
            })

            total_samples += dataset_samples
            domain_stats["total_samples"] = total_samples

            print(f"✅ {spec.name}: {dataset_samples} samples added")

        except Exception as e:
            print(f"❌ Failed to download {dataset_spec}: {e}")
            continue

    # Save domain metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(domain_stats, f, indent=2)

    print(f"✅ {domain} corpus: {total_samples} samples")
    return domain_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Download domain-specific HF corpora")
    parser.add_argument(
        "--domain",
        required=True,
        choices=["physics", "math", "code", "chess", "general"],
        help="Domain to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/hf_corpus"),
        help="Output directory",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=50000,
        help="Target samples per domain",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Minimum characters per sample",
    )

    args = parser.parse_args()

    # Get HF token
    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable or --hf-token required")

    # Set HF_TOKEN env var for the corpus builder
    os.environ["HF_TOKEN"] = token

    print(f"🚀 Downloading {args.domain} corpus")
    print(f"📁 Output: {args.output_dir / args.domain}")
    print(f"🎯 Target: {args.target_samples} samples")

    stats = download_domain_corpus(
        domain=args.domain,
        output_dir=args.output_dir / args.domain,
        token=token,
        target_samples=args.target_samples,
        min_chars=args.min_chars
    )

    print(f"✅ Completed: {stats['total_samples']} total samples")


if __name__ == "__main__":
    main()
