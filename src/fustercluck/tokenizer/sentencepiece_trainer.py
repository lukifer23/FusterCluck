"""Utilities to train SentencePiece tokenizers for FusterCluck."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

import sentencepiece as spm

LOGGER = logging.getLogger(__name__)


def train_sentencepiece(
    corpus_files: Sequence[Path],
    model_prefix: Path,
    vocab_size: int = 50000,
    character_coverage: float = 0.9995,
    user_defined_symbols: Iterable[str] = (),
    normalization_rule_name: str = "identity",
    input_sentence_size: int = 20_000_000,
    shuffle_input_sentence: bool = True,
) -> Path:
    """Train a SentencePiece tokenizer from the given corpus files."""

    if not corpus_files:
        raise ValueError("At least one corpus file is required")
    for file in corpus_files:
        if not file.exists():
            raise FileNotFoundError(file)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    extra_symbols = list(dict.fromkeys(user_defined_symbols))  # remove duplicates while preserving order
    train_cmd = {
        "input": ",".join(str(path) for path in corpus_files),
        "model_prefix": str(model_prefix),
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "hard_vocab_limit": False,
        "model_type": "unigram",
        "user_defined_symbols": ",".join(extra_symbols),
        "normalization_rule_name": normalization_rule_name,
        "input_sentence_size": input_sentence_size,
        "shuffle_input_sentence": int(shuffle_input_sentence),
        "max_sentence_length": 4096,
        "num_threads": max(1, os.cpu_count() or 1 // 2),
    }
    LOGGER.info("Training SentencePiece model: %s", train_cmd)
    spm.SentencePieceTrainer.train(**train_cmd)
    model_path = model_prefix.with_suffix(".model")
    if not model_path.exists():
        raise RuntimeError("SentencePiece training failed; model file missing")
    LOGGER.info("Saved SentencePiece model to %s", model_path)
    return model_path
