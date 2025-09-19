#!/usr/bin/env python3
"""Create a tarball bundle of training assets for easy transfer between machines."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Iterable

DEFAULT_PATHS = [
    "configs",
    "artifacts/tokenizer",
    "data/raw",
    "data/tokenized",
    "scripts",
    "Plan.md",
    "README.md",
]


def iter_existing(paths: Iterable[Path]) -> list[Path]:
    resolved = []
    for path in paths:
        if path.exists():
            resolved.append(path)
        else:
            print(f"[warn] Skipping missing path: {path}")
    return resolved


def create_bundle(output: Path, include: list[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        for path in include:
            arcname = path.relative_to(Path.cwd()) if path.is_absolute() else path
            print(f"Adding {path} -> {arcname}")
            tar.add(path, arcname=str(arcname))
    print(f"Bundle written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/bundles/training_bundle.tar.gz"))
    parser.add_argument(
        "--paths",
        nargs="*",
        type=Path,
        default=[Path(p) for p in DEFAULT_PATHS],
        help="Additional paths to include in the bundle",
    )
    args = parser.parse_args()

    include = iter_existing(args.paths)
    if not include:
        raise SystemExit("No valid paths to bundle")

    create_bundle(args.output, include)


if __name__ == "__main__":
    main()
