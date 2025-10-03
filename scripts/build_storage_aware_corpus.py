#!/usr/bin/env python3
"""Storage-aware corpus building that respects SSD constraints."""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import psutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def get_disk_usage(path: Path) -> Tuple[float, float, float]:
    """Get disk usage for the filesystem containing path. Returns (used_gb, total_gb, available_gb)."""
    # If path doesn't exist, use the parent directory
    check_path = path if path.exists() else path.parent
    if not check_path.exists():
        check_path = Path.home()  # Fallback to home directory

    stat = os.statvfs(check_path)
    total_bytes = stat.f_bsize * stat.f_blocks
    available_bytes = stat.f_bsize * stat.f_bavail
    used_bytes = total_bytes - available_bytes

    total_gb = total_bytes / (1024**3)
    used_gb = used_bytes / (1024**3)
    available_gb = available_bytes / (1024**3)

    return used_gb, total_gb, available_gb


def estimate_token_count(text: str) -> int:
    """Rough estimate of tokens (1 token ≈ 4 chars)."""
    return max(len(text.split()), len(text) // 4)


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024**2)


def build_domain_corpus(
    domain: str,
    source_dirs: List[Path],
    output_dir: Path,
    max_disk_usage_percent: float = 80.0,
    target_tokens: int = 500_000_000,
    chunk_size_mb: int = 100
) -> Dict[str, int]:
    """
    Build corpus for a domain with storage awareness.

    Args:
        domain: Domain name (physics, math, etc.)
        source_dirs: List of directories containing source files
        output_dir: Output directory for processed corpus
        max_disk_usage_percent: Maximum disk usage percentage before stopping
        target_tokens: Target number of tokens for this domain
        chunk_size_mb: Process files in chunks of this size

    Returns:
        Statistics dictionary
    """

    domain_output = output_dir / domain
    domain_output.mkdir(parents=True, exist_ok=True)

    stats = {
        "domain": domain,
        "files_processed": 0,
        "total_tokens": 0,
        "total_size_mb": 0.0,
        "chunks_created": 0
    }

    # Collect all source files
    all_files = []
    for source_dir in source_dirs:
        if source_dir.exists():
            for ext in ['*.txt', '*.py', '*.pgn', '*.jsonl']:
                all_files.extend(source_dir.glob(f"**/{ext}"))

    logger.info(f"📂 Found {len(all_files)} source files for {domain}")

    if not all_files:
        logger.warning(f"⚠️  No source files found for {domain}")
        return stats

    # Sort files by size (process smaller files first to get quick wins)
    all_files.sort(key=lambda f: f.stat().st_size)

    current_chunk = []
    current_chunk_size = 0
    chunk_counter = 0

    for file_path in all_files:
        try:
            # Check disk usage before processing
            used_gb, total_gb, available_gb = get_disk_usage(output_dir)
            disk_usage_percent = (used_gb / total_gb) * 100

            if disk_usage_percent > max_disk_usage_percent:
                logger.warning(".1f"
                               f"Stopping {domain} processing to preserve disk space")
                break

            file_size_mb = get_file_size_mb(file_path)

            # Skip files that are too large
            if file_size_mb > chunk_size_mb * 2:
                logger.warning(f"⚠️  Skipping large file: {file_path} ({file_size_mb:.1f}MB)")
                continue

            # Read and process file
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"⚠️  Failed to read {file_path}: {e}")
                continue

            if not content.strip():
                continue

            # Estimate tokens
            token_count = estimate_token_count(content)
            current_chunk.append(content)
            current_chunk_size += file_size_mb
            stats["files_processed"] += 1
            stats["total_tokens"] += token_count
            stats["total_size_mb"] += file_size_mb

            # Write chunk when it reaches target size or token count
            should_write = (
                current_chunk_size >= chunk_size_mb or
                stats["total_tokens"] >= target_tokens
            )

            if should_write and current_chunk:
                chunk_file = domain_output / "02d"
                try:
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        for text in current_chunk:
                            f.write(text)
                            f.write('\n\n')  # Separator between documents

                    chunk_token_count = sum(estimate_token_count(text) for text in current_chunk)
                    chunk_size_mb_actual = get_file_size_mb(chunk_file)

                    logger.info(f"💾 Wrote chunk {chunk_counter}: {chunk_file} "
                               f"({chunk_size_mb_actual:.1f}MB, {chunk_token_count:,} tokens)")

                    stats["chunks_created"] += 1
                    chunk_counter += 1

                except Exception as e:
                    logger.error(f"❌ Failed to write chunk {chunk_counter}: {e}")

                current_chunk = []
                current_chunk_size = 0

            # Check if we've reached the target
            if stats["total_tokens"] >= target_tokens:
                logger.info(f"🎯 Reached target of {target_tokens:,} tokens for {domain}")
                break

            # Progress logging
            if stats["files_processed"] % 100 == 0:
                logger.info(f"📊 {domain}: {stats['files_processed']} files, "
                           f"{stats['total_tokens']:,} tokens, {stats['total_size_mb']:.1f}MB")

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            continue

    # Write final chunk if any remains
    if current_chunk:
        chunk_file = domain_output / "02d"
        try:
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for text in current_chunk:
                    f.write(text)
                    f.write('\n\n')

            chunk_token_count = sum(estimate_token_count(text) for text in current_chunk)
            chunk_size_mb_actual = get_file_size_mb(chunk_file)

            logger.info(f"💾 Wrote final chunk {chunk_counter}: {chunk_file} "
                       f"({chunk_size_mb_actual:.1f}MB, {chunk_token_count:,} tokens)")

            stats["chunks_created"] += 1

        except Exception as e:
            logger.error(f"❌ Failed to write final chunk: {e}")

    logger.info(f"✅ {domain} corpus complete: {stats['files_processed']} files → "
               f"{stats['chunks_created']} chunks, {stats['total_tokens']:,} tokens")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build storage-aware corpus")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/domains"),
                       help="Source directory containing domain subfolders")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"),
                       help="Output directory for processed corpus")
    parser.add_argument("--max-disk-percent", type=float, default=80.0,
                       help="Maximum disk usage percentage")
    parser.add_argument("--target-tokens-per-domain", type=int, default=500_000_000,
                       help="Target tokens per domain")
    parser.add_argument("--chunk-size-mb", type=int, default=100,
                       help="Chunk size in MB")
    parser.add_argument("--domains", nargs="+",
                       choices=["physics", "math", "code", "chess", "general"],
                       default=["physics", "math", "code", "chess", "general"],
                       help="Domains to process")

    args = parser.parse_args()

    # Check initial disk usage
    used_gb, total_gb, available_gb = get_disk_usage(args.output_dir)
    disk_usage_percent = (used_gb / total_gb) * 100

    logger.info("💽 Disk status: "
               ".1f"
               ".1f"
               ".1f")

    if disk_usage_percent > args.max_disk_percent:
        logger.error(".1f"
                     "Aborting to prevent disk space issues.")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each domain
    total_stats = {}

    for domain in args.domains:
        source_dirs = [args.source_dir / domain]

        # Also include expanded data if available
        expanded_dir = Path("data/raw/domains_expanded") / domain
        if expanded_dir.exists():
            source_dirs.append(expanded_dir)

        logger.info(f"🏗️  Building corpus for {domain}...")

        domain_stats = build_domain_corpus(
            domain=domain,
            source_dirs=source_dirs,
            output_dir=args.output_dir,
            max_disk_usage_percent=args.max_disk_percent,
            target_tokens=args.target_tokens_per_domain,
            chunk_size_mb=args.chunk_size_mb
        )

        total_stats[domain] = domain_stats

    # Final summary
    logger.info("=" * 60)
    logger.info("📊 CORPUS BUILDING SUMMARY")
    logger.info("=" * 60)

    grand_total_tokens = 0
    grand_total_files = 0
    grand_total_chunks = 0

    for domain, stats in total_stats.items():
        logger.info(f"{domain.upper():8}: {stats['files_processed']:4d} files → "
                   f"{stats['chunks_created']:2d} chunks, "
                   f"{stats['total_tokens']:8,d} tokens, "
                   ".1f")

        grand_total_files += stats['files_processed']
        grand_total_tokens += stats['total_tokens']
        grand_total_chunks += stats['chunks_created']

    logger.info("-" * 60)
    logger.info(f"{'TOTAL':8}: {grand_total_files:4d} files → "
               f"{grand_total_chunks:2d} chunks, "
               f"{grand_total_tokens:8,d} tokens")

    # Final disk check
    final_used_gb, final_total_gb, final_available_gb = get_disk_usage(args.output_dir)
    final_disk_usage_percent = (final_used_gb / final_total_gb) * 100

    logger.info(".1f")


if __name__ == "__main__":
    main()
