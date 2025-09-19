#!/usr/bin/env python3
"""Generate synthetic OCR samples and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fustercluck.data.ocr_synthetic import SynthOCRConfig, SyntheticOCRGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/synth_ocr"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--font", action="append", type=Path, help="Additional font file paths")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fonts = [Path(f) for f in args.font] if args.font else None
    cfg = SynthOCRConfig(width=args.width, height=args.height, noise_level=args.noise, fonts=fonts)
    generator = SyntheticOCRGenerator(cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx in range(args.count):
        image, strings = generator.generate()
        image_path = args.output / f"sample_{idx:06d}.png"
        image.save(image_path, format="PNG")
        manifest.append({"image": image_path.name, "strings": strings})
    manifest_path = args.output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in manifest:
            handle.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
