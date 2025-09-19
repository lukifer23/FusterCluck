#!/usr/bin/env python3
"""Build Stage 0 corpus from domain-specific manifests."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import deque
from itertools import cycle
from pathlib import Path
from typing import Iterable, Iterator, List

LOGGER = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    sources = data.get("sources")
    if not sources:
        raise ValueError("Manifest must include a 'sources' list")
    normalized = []
    weight_total = 0.0
    for entry in sources:
        weight = float(entry.get("weight", 0.0))
        if weight <= 0.0:
            raise ValueError(f"Source missing positive weight: {entry}")
        weight_total += weight
        paths = entry.get("paths") or [entry.get("path")]
        if not paths:
            raise ValueError(f"Source entry requires 'path' or 'paths': {entry}")
        resolved: List[Path] = []
        for path in paths:
            if path is None:
                continue
            matches = list(Path().glob(path)) if any(ch in path for ch in "*?[") else [Path(path)]
            if not matches:
                LOGGER.warning("No files matched %s", path)
            resolved.extend(matches)
        if not resolved:
            raise FileNotFoundError(f"No files found for source: {entry}")
        normalized.append({
            "name": entry.get("name", resolved[0].stem),
            "paths": resolved,
            "weight": weight,
        })
    return normalized


def iter_lines(paths: Iterable[Path]) -> Iterator[str]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield line.rstrip("\n")


def estimate_tokens(text: str) -> int:
    return max(len(text.split()), 1)


def build_corpus(manifest: Path, output: Path, target_tokens: int) -> dict:
    sources = load_manifest(manifest)
    weight_total = sum(src["weight"] for src in sources)
    token_budget = {src["name"]: int(target_tokens * (src["weight"] / weight_total)) for src in sources}
    LOGGER.info("Token budget per source: %s", token_budget)

    queues: dict[str, deque[str]] = {}
    for src in sources:
        lines = list(iter_lines(src["paths"]))
        if not lines:
            raise RuntimeError(f"Source {src['name']} is empty")
        random.shuffle(lines)
        queues[src["name"]] = deque(lines)
        LOGGER.info("Loaded %d lines for %s", len(lines), src["name"])

    seen = set()
    written_lines: List[str] = []
    approx_tokens = 0
    domain_tokens = {name: 0 for name in queues}

    for src in cycle(sources):
        name = src["name"]
        if domain_tokens[name] >= token_budget[name]:
            if all(domain_tokens[n] >= token_budget[n] for n in domain_tokens):
                break
            continue
        queue = queues[name]
        if not queue:
            LOGGER.warning("Source %s exhausted before hitting target", name)
            domain_tokens[name] = token_budget[name]
            continue
        line = queue.popleft()
        if line in seen:
            continue
        seen.add(line)
        tokens = estimate_tokens(line)
        approx_tokens += tokens
        domain_tokens[name] += tokens
        written_lines.append(line)
        if approx_tokens >= target_tokens:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for line in written_lines:
            handle.write(line + "\n")

    manifest_data = {
        "manifest": str(manifest),
        "target_tokens": target_tokens,
        "approx_tokens": approx_tokens,
        "domain_tokens": domain_tokens,
        "lines": len(written_lines),
        "output": str(output),
    }
    manifest_path = output.with_suffix(".json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, indent=2)
    LOGGER.info("Wrote %d lines (~%d tokens)", len(written_lines), approx_tokens)
    return manifest_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="YAML manifest describing domain sources")
    parser.add_argument("--output", type=Path, default=Path("data/raw/stage0_domain.txt"))
    parser.add_argument("--target-tokens", type=int, default=50_000_000)
    parser.add_argument("--log", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()), format="[%(levelname)s] %(message)s")
    build_corpus(args.manifest, args.output, args.target_tokens)


if __name__ == "__main__":
    main()
