#!/usr/bin/env python3
"""Deduplicate corpus using MinHash and content-based filtering."""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class MinHashDeduper:
    """MinHash-based deduplication for text content."""

    def __init__(self, num_hashes: int = 128, shingle_size: int = 5):
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        self.hashes: List[int] = []
        self._init_hashes()

    def _init_hashes(self):
        """Initialize hash functions with random seeds."""
        np.random.seed(42)  # For reproducibility
        self.hash_seeds = np.random.randint(0, 2**32, size=self.num_hashes, dtype=np.uint32)

    def _get_shingles(self, text: str) -> List[str]:
        """Extract shingles from text."""
        words = text.lower().split()
        if len(words) < self.shingle_size:
            return [' '.join(words)]  # Return whole text if too short

        shingles = []
        for i in range(len(words) - self.shingle_size + 1):
            shingle = ' '.join(words[i:i + self.shingle_size])
            shingles.append(shingle)
        return shingles

    def _hash_shingle(self, shingle: str, seed: int) -> int:
        """Hash a shingle with a specific seed."""
        combined = f"{shingle}:{seed}".encode('utf-8')
        return int(hashlib.md5(combined).hexdigest(), 16)

    def get_signature(self, text: str) -> np.ndarray:
        """Get MinHash signature for text."""
        shingles = self._get_shingles(text)
        if not shingles:
            return np.full(self.num_hashes, 2**32 - 1, dtype=np.uint32)

        signature = np.full(self.num_hashes, 2**32 - 1, dtype=np.uint32)

        for shingle in shingles:
            for i, seed in enumerate(self.hash_seeds):
                h = self._hash_shingle(shingle, seed)
                signature[i] = min(signature[i], h)

        return signature

    def jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Calculate Jaccard similarity between two signatures."""
        if len(sig1) != len(sig2):
            return 0.0
        matches = np.sum(sig1 == sig2)
        return matches / len(sig1)


def content_hash(text: str) -> str:
    """Simple content hash for exact duplicates."""
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()


def deduplicate_files(
    input_dir: Path,
    output_dir: Path,
    similarity_threshold: float = 0.9,
    min_length: int = 50
) -> Dict[str, int]:
    """
    Deduplicate text files using content hashing and MinHash.

    Args:
        input_dir: Directory containing chunk files
        output_dir: Output directory for deduplicated files
        similarity_threshold: MinHash similarity threshold for near-duplicates
        min_length: Minimum text length to consider

    Returns:
        Statistics dictionary
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all text files (chunk files from storage-aware builder)
    input_files = []
    for pattern in ["chunk_*.txt", "*.txt", "*"]:
        input_files = list(input_dir.glob(pattern))
        if input_files:
            break

    if not input_files:
        logger.warning(f"⚠️  No text files found in {input_dir}")
        return {
            "files_processed": 0,
            "total_lines": 0,
            "unique_lines": 0,
            "exact_duplicates": 0,
            "near_duplicates": 0
        }

    logger.info(f"📂 Found {len(input_files)} input files")

    # Initialize deduplication
    minhash = MinHashDeduper(num_hashes=64, shingle_size=5)  # Smaller for speed
    seen_hashes: Set[str] = set()
    seen_signatures: List[Tuple[np.ndarray, str]] = []  # (signature, content_hash)

    stats = {
        "files_processed": 0,
        "total_lines": 0,
        "unique_lines": 0,
        "exact_duplicates": 0,
        "near_duplicates": 0
    }

    # Process each file
    for input_file in input_files:
        logger.info(f"🔍 Processing {input_file.name}")

        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"❌ Failed to read {input_file}: {e}")
            continue

        lines = content.split('\n\n')  # Split on document separator
        logger.info(f"  Found {len(lines)} documents")

        deduplicated_lines = []

        for line in lines:
            line = line.strip()
            if not line or len(line) < min_length:
                continue

            stats["total_lines"] += 1

            # Exact duplicate check
            content_h = content_hash(line)
            if content_h in seen_hashes:
                stats["exact_duplicates"] += 1
                continue

            # Near-duplicate check using MinHash
            signature = minhash.get_signature(line)
            is_near_duplicate = False

            for existing_sig, existing_hash in seen_signatures:
                similarity = minhash.jaccard_similarity(signature, existing_sig)
                if similarity >= similarity_threshold:
                    is_near_duplicate = True
                    stats["near_duplicates"] += 1
                    break

            if not is_near_duplicate:
                deduplicated_lines.append(line)
                seen_hashes.add(content_h)
                seen_signatures.append((signature, content_h))
                stats["unique_lines"] += 1

        # Write deduplicated content
        if deduplicated_lines:
            output_file = output_dir / f"dedup_{input_file.name}"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n\n'.join(deduplicated_lines))

                file_size_mb = output_file.stat().st_size / (1024**2)
                logger.info(f"💾 Wrote {output_file.name}: {len(deduplicated_lines)} docs, {file_size_mb:.1f}MB")

            except Exception as e:
                logger.error(f"❌ Failed to write {output_file}: {e}")

        stats["files_processed"] += 1

        # Memory management: limit stored signatures
        if len(seen_signatures) > 10000:
            # Keep most recent signatures
            seen_signatures = seen_signatures[-5000:]
            logger.info("🧹 Cleaned up signature cache")

    return stats


def deduplicate_domain(
    domain: str,
    processed_dir: Path,
    output_dir: Path,
    similarity_threshold: float = 0.9
) -> Dict[str, int]:
    """Deduplicate all chunks for a specific domain."""

    domain_input = processed_dir / domain
    domain_output = output_dir / domain

    if not domain_input.exists():
        logger.warning(f"⚠️  Domain directory not found: {domain_input}")
        return {}

    logger.info(f"🔄 Deduplicating domain: {domain}")

    stats = deduplicate_files(
        input_dir=domain_input,
        output_dir=domain_output,
        similarity_threshold=similarity_threshold
    )

    if stats:
        logger.info(f"✅ {domain}: {stats['total_lines']} → {stats['unique_lines']} unique "
                   f"({stats['exact_duplicates']} exact + {stats['near_duplicates']} near duplicates removed)")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate corpus using MinHash")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"),
                       help="Input directory with processed chunks")
    parser.add_argument("--output-dir", type=Path, default=Path("data/deduped"),
                       help="Output directory for deduplicated files")
    parser.add_argument("--similarity-threshold", type=float, default=0.9,
                       help="MinHash similarity threshold for near-duplicates")
    parser.add_argument("--domains", nargs="+",
                       choices=["physics", "math", "code", "chess", "general"],
                       default=["physics", "math", "code", "chess", "general"],
                       help="Domains to deduplicate")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🧹 Starting corpus deduplication")
    logger.info(f"📁 Input: {args.input_dir}")
    logger.info(f"📁 Output: {args.output_dir}")
    logger.info(f"🎯 Similarity threshold: {args.similarity_threshold}")

    total_stats = {}

    for domain in args.domains:
        domain_stats = deduplicate_domain(
            domain=domain,
            processed_dir=args.input_dir,
            output_dir=args.output_dir,
            similarity_threshold=args.similarity_threshold
        )
        total_stats[domain] = domain_stats

    # Summary
    logger.info("=" * 60)
    logger.info("📊 DEDUPLICATION SUMMARY")
    logger.info("=" * 60)

    grand_total = 0
    grand_unique = 0
    grand_exact_dupes = 0
    grand_near_dupes = 0

    for domain, stats in total_stats.items():
        if stats:
            logger.info(f"{domain.upper():8}: {stats['total_lines']:6,d} → {stats['unique_lines']:6,d} "
                       f"(-{stats['exact_duplicates']:4,d} exact, -{stats['near_duplicates']:4,d} near)")

            grand_total += stats['total_lines']
            grand_unique += stats['unique_lines']
            grand_exact_dupes += stats['exact_duplicates']
            grand_near_dupes += stats['near_duplicates']

    if grand_total > 0:
        dedupe_rate = (grand_exact_dupes + grand_near_dupes) / grand_total * 100
        logger.info("-" * 60)
        logger.info(f"{'TOTAL':8}: {grand_total:6,d} → {grand_unique:6,d} "
                   f"({dedupe_rate:.1f}% duplicates removed)")


if __name__ == "__main__":
    main()
