#!/usr/bin/env python3
"""Train SentencePiece tokenizer for the text-only FusterCluck stack."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fustercluck.tokenizer.sentencepiece_trainer import train_sentencepiece

EXTRA_SYMBOLS = [
    "<reasoning>",
    "</reasoning>",
    "<tool>",
    "<json>",
    "<sys>",
    "<scratchpad>",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="+", type=Path, help="Path(s) to plain text files")
    parser.add_argument("--output", type=Path, default=Path("artifacts/tokenizer/fustercluck"))
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument("--coverage", type=float, default=0.9995)
    parser.add_argument("--sentence-sample", type=int, default=20_000_000)
    parser.add_argument(
        "--normalization",
        type=str,
        default="identity",
        help="SentencePiece normalization rule (identity, nfkc_cf, etc.)",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    train_sentencepiece(
        corpus_files=args.corpus,
        model_prefix=args.output,
        vocab_size=args.vocab_size,
        character_coverage=args.coverage,
        user_defined_symbols=EXTRA_SYMBOLS,
        normalization_rule_name=args.normalization,
        input_sentence_size=args.sentence_sample,
    )


if __name__ == "__main__":
    main()
