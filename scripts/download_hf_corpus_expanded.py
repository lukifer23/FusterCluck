#!/usr/bin/env python3
"""Download large Hugging Face datasets for expanded corpus."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def download_hf_datasets(output_dir: Path, target_tokens_per_domain: int = 500_000_000) -> None:
    """Download large HF datasets to expand corpus significantly."""

    # Dataset configurations for different domains - using working datasets
    dataset_configs = {
        "physics": [
            {
                "name": "wikipedia",
                "config": "20220301.en",
                "split": "train",
                "text_field": "text",
                "sample_limit": 50_000,
                "target_tokens": int(target_tokens_per_domain * 0.35),  # 35% physics
            }
        ],
        "math": [
            {
                "name": "wikipedia",
                "config": "20220301.en",
                "split": "train",
                "text_field": "text",
                "sample_limit": 40_000,
                "target_tokens": int(target_tokens_per_domain * 0.30),  # 30% math
            }
        ],
        "code": [
            {
                "name": "code_search_net",
                "config": "python",
                "split": "train",
                "text_field": "func_code_string",
                "sample_limit": 100_000,
                "target_tokens": int(target_tokens_per_domain * 0.20),  # 20% code
            }
        ],
        "general": [
            {
                "name": "wikipedia",
                "config": "20220301.en",
                "split": "train",
                "text_field": "text",
                "sample_limit": 30_000,
                "target_tokens": int(target_tokens_per_domain * 0.15),  # 15% general
            }
        ]
    }

    for domain, configs in dataset_configs.items():
        domain_dir = output_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📥 Downloading datasets for domain: {domain}")

        for config in configs:
            try:
                logger.info(f"  Loading {config['name']} ({config['sample_limit']:,} samples)")

                # Load dataset
                if config["config"]:
                    dataset_path = f"{config['name']}/{config['config']}"
                else:
                    dataset_path = config["name"]

                dataset = load_dataset(dataset_path, split=config["split"], streaming=True)

                # Download and save samples
                output_file = domain_dir / f"{config['name'].replace('/', '_')}_{config['split']}.txt"
                total_tokens = 0
                samples_saved = 0

                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in tqdm(dataset, desc=f"Downloading {domain}", unit="samples"):
                        text = sample[config["text_field"]]
                        if text and len(text.strip()) > 50:  # Minimum length filter
                            f.write(text.strip() + '\n\n')
                            # Rough token estimate (1 token ≈ 4 chars)
                            total_tokens += len(text) // 4
                            samples_saved += 1

                            if samples_saved >= config["sample_limit"]:
                                break

                            if total_tokens >= config["target_tokens"]:
                                break

                logger.info(f"  ✅ Saved {samples_saved:,} samples ({total_tokens:,} tokens) to {output_file}")

            except Exception as e:
                logger.error(f"  ❌ Failed to download {config['name']}: {e}")
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Download expanded corpus from Hugging Face")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/domains_expanded"))
    parser.add_argument("--target-tokens-per-domain", type=int, default=500_000_000,
                       help="Target tokens per domain (default: 500M)")

    args = parser.parse_args()

    logger.info("🚀 Starting expanded corpus download")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"🎯 Target tokens per domain: {args.target_tokens_per_domain:,}")

    download_hf_datasets(args.output_dir, args.target_tokens_per_domain)

    logger.info("✅ Corpus expansion complete!")


if __name__ == "__main__":
    main()
