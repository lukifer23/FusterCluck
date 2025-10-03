#!/usr/bin/env python3
"""Chunk existing domain text files into fixed word windows to increase unique samples."""

from __future__ import annotations

import argparse
from pathlib import Path


def chunk_file(path: Path, output_dir: Path, *, chunk_words: int, min_words: int, suffix: str) -> Path:
    text = path.read_text(encoding="utf-8", errors="ignore")
    words = text.split()
    if len(words) <= min_words:
        return path
    chunks = []
    for start in range(0, len(words), chunk_words):
        window = words[start : start + chunk_words]
        if len(window) < min_words:
            continue
        chunks.append(" ".join(window))
    if not chunks:
        return path
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{path.stem}{suffix}{path.suffix}"
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.strip() + "\n")
    return output


def process_domain(directory: Path, output_dir: Path, *, extensions: tuple[str, ...], chunk_words: int, min_words: int, suffix: str) -> list[Path]:
    generated: list[Path] = []
    for file in directory.rglob("*"):
        if file.suffix.lower() not in extensions:
            continue
        if file.parent == output_dir or suffix.strip("_") in file.stem:
            continue
        output = chunk_file(file, output_dir, chunk_words=chunk_words, min_words=min_words, suffix=suffix)
        if output != file:
            generated.append(output)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root directory containing domain subfolders")
    parser.add_argument("--chunk-words", type=int, default=256, help="Number of words per chunk")
    parser.add_argument("--min-words", type=int, default=64, help="Discard chunks shorter than this many words")
    parser.add_argument("--suffix", type=str, default="_chunked", help="Suffix added to new files")
    parser.add_argument("--domains", type=str, default="", help="Comma-separated list of domains to process (defaults to physics,math,general)")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional directory to store chunked files (defaults beside originals)")
    args = parser.parse_args()
    generated_total = []
    domains = args.domains.split(",") if args.domains else ["physics", "math", "general"]
    for domain in [d.strip() for d in domains if d.strip()]:
        domain_dir = args.root / domain
        if not domain_dir.exists():
            continue
        output_dir = domain_dir if args.output_root is None else args.output_root / domain
        generated = process_domain(domain_dir, output_dir, extensions=(".txt",), chunk_words=args.chunk_words, min_words=args.min_words, suffix=args.suffix)
        generated_total.extend(generated)
        if generated:
            print(f"Generated {len(generated)} chunk files in {output_dir}")
    if not generated_total:
        print("No chunk files generated (all files too small or already chunked).")


if __name__ == "__main__":
    main()
