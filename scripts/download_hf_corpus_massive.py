#!/usr/bin/env python3
"""Massive-scale download script for 6B token pre-training corpus."""

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
    """Download massive corpus for a specific domain using multiple sources."""
    
    # AGGRESSIVE: 8 STEM domains scaled to hit 3.5B tokens
    domain_datasets = {
        "physics": [
            # Massive physics content for 3.5B target
            "HuggingFaceFW/fineweb-edu:sample-10BT:train:text::800000",   # 800k samples
            "HuggingFaceFW/fineweb-edu:sample-100BT:train:text::600000", # 600k samples
            "HuggingFaceFW/fineweb-edu:sample-350BT:train:text::400000", # 400k samples

            # Different time periods for diversity
            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-05:train:text::300000", # 300k
            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-13:train:text::300000", # 300k
            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-21:train:text::300000", # 300k
            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-26:train:text::300000", # 300k

            # General web with physics content
            "HuggingFaceFW/fineweb:sample-10BT:train:text::500000",     # 500k
            "HuggingFaceFW/fineweb:sample-100BT:train:text::400000",    # 400k
        ],
        "math": [
            "HuggingFaceFW/fineweb-edu:sample-10BT:train:text::800000",   # 800k
            "HuggingFaceFW/fineweb-edu:sample-100BT:train:text::600000", # 600k
            "HuggingFaceFW/fineweb-edu:sample-350BT:train:text::400000", # 400k

            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-08:train:text::300000", # 300k
            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-18:train:text::300000", # 300k
            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-26:train:text::300000", # 300k
            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-05:train:text::300000", # 300k

            "HuggingFaceFW/fineweb:sample-10BT:train:text::500000",     # 500k
            "HuggingFaceFW/fineweb:sample-100BT:train:text::400000",    # 400k
        ],
        "code": [
            # Massive programming content - prioritized
            "HuggingFaceFW/fineweb-edu:sample-10BT:train:text::1000000",  # 1M educational
            "HuggingFaceFW/fineweb-edu:sample-100BT:train:text::800000", # 800k educational
            "HuggingFaceFW/fineweb-edu:sample-350BT:train:text::600000", # 600k educational

            "HuggingFaceFW/fineweb:sample-10BT:train:text::1500000",     # 1.5M general
            "HuggingFaceFW/fineweb:sample-100BT:train:text::1000000",   # 1M general
            "HuggingFaceFW/fineweb:sample-350BT:train:text::800000",    # 800k general

            # Different snapshots for code evolution
            "HuggingFaceFW/fineweb:CC-MAIN-2025-05:train:text::400000", # 400k
            "HuggingFaceFW/fineweb:CC-MAIN-2025-13:train:text::400000", # 400k
            "HuggingFaceFW/fineweb:CC-MAIN-2025-18:train:text::400000", # 400k
        ],
        "chemistry": [
            # Chemistry domain - scaled up
            "HuggingFaceFW/fineweb-edu:sample-10BT:train:text::600000",   # 600k educational
            "HuggingFaceFW/fineweb-edu:sample-100BT:train:text::400000", # 400k educational
            "HuggingFaceFW/fineweb-edu:sample-350BT:train:text::300000", # 300k educational

            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-08:train:text::250000", # 250k
            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-18:train:text::250000", # 250k
            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-26:train:text::250000", # 250k

            "HuggingFaceFW/fineweb:sample-10BT:train:text::300000",     # 300k general
            "HuggingFaceFW/fineweb:sample-100BT:train:text::200000",    # 200k general
        ],
        "biology": [
            # Biology domain - scaled up
            "HuggingFaceFW/fineweb-edu:sample-10BT:train:text::600000",   # 600k educational
            "HuggingFaceFW/fineweb-edu:sample-100BT:train:text::400000", # 400k educational
            "HuggingFaceFW/fineweb-edu:sample-350BT:train:text::300000", # 300k educational

            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-05:train:text::250000", # 250k
            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-13:train:text::250000", # 250k
            "HuggingFaceFW/fineweb-edu:CC-MAIN-2025-21:train:text::250000", # 250k

            "HuggingFaceFW/fineweb:sample-10BT:train:text::300000",     # 300k general
            "HuggingFaceFW/fineweb:sample-100BT:train:text::200000",    # 200k general
        ],
        "history": [
            # History domain - scaled up
            "HuggingFaceFW/fineweb-edu:sample-10BT:train:text::400000",   # 400k educational
            "HuggingFaceFW/fineweb-edu:sample-100BT:train:text::300000", # 300k educational

            "HuggingFaceFW/fineweb:sample-10BT:train:text::800000",      # 800k general
            "HuggingFaceFW/fineweb:sample-100BT:train:text::600000",    # 600k general
            "HuggingFaceFW/fineweb:sample-350BT:train:text::400000",    # 400k general

            # Historical content from different periods
            "HuggingFaceFW/fineweb:CC-MAIN-2025-05:train:text::200000", # 200k
            "HuggingFaceFW/fineweb:CC-MAIN-2025-08:train:text::200000", # 200k
            "HuggingFaceFW/fineweb:CC-MAIN-2025-13:train:text::200000", # 200k
        ],
        "general": [
            # General content - scaled up for breadth
            "HuggingFaceFW/fineweb:sample-10BT:train:text::2000000",     # 2M samples
            "HuggingFaceFW/fineweb:sample-100BT:train:text::1500000",   # 1.5M samples
            "HuggingFaceFW/fineweb:sample-350BT:train:text::1000000",   # 1M samples

            # Multiple time periods for temporal diversity
            "HuggingFaceFW/fineweb:CC-MAIN-2025-05:train:text::500000", # 500k
            "HuggingFaceFW/fineweb:CC-MAIN-2025-08:train:text::500000", # 500k
            "HuggingFaceFW/fineweb:CC-MAIN-2025-13:train:text::500000", # 500k
            "HuggingFaceFW/fineweb:CC-MAIN-2025-18:train:text::500000", # 500k
            "HuggingFaceFW/fineweb:CC-MAIN-2025-21:train:text::500000", # 500k
            "HuggingFaceFW/fineweb:CC-MAIN-2025-26:train:text::500000", # 500k
        ],
        "chess": [
            # Chess content (smaller scale since it's specialized)
            "HuggingFaceFW/fineweb-edu:sample-10BT:train:text::100000",  # 100k educational
            "HuggingFaceFW/fineweb-edu:sample-100BT:train:text::80000",  # 80k educational
            "HuggingFaceFW/fineweb:sample-10BT:train:text::200000",     # 200k general (chess content)
            "HuggingFaceFW/fineweb:sample-100BT:train:text::100000",    # 100k general
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
                "config": spec.config,
                "samples": dataset_samples
            })

            total_samples += dataset_samples
            domain_stats["total_samples"] = total_samples

            print(f"✅ {spec.name}:{spec.config or 'default'}: {dataset_samples} samples added")

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
    parser = argparse.ArgumentParser(description="Download massive-scale HF corpora for 3.5B token STEM training")
    parser.add_argument(
        "--domain",
        required=True,
        choices=["physics", "math", "code", "chemistry", "biology", "history", "chess", "general"],
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
        default=1000000,  # 1M default per domain
        help="Target samples per domain (may be exceeded by dataset limits)",
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

    print(f"🚀 Downloading MASSIVE {args.domain} corpus for 6B token training")
    print(f"📁 Output: {args.output_dir / args.domain}")
    print(f"🎯 Target: {args.target_samples:,} samples per domain")

    stats = download_domain_corpus(
        domain=args.domain,
        output_dir=args.output_dir / args.domain,
        token=token,
        target_samples=args.target_samples,
        min_chars=args.min_chars
    )

    print(f"✅ Completed: {stats['total_samples']:,} total samples")
    print("💡 This should provide sufficient data for 3.5B token training when combined across all domains")


if __name__ == "__main__":
    main()
