#!/usr/bin/env python3
"""Materialize production Stage 1/2 corpora directly from Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf

from fustercluck.data.corpus_builder import (
    DatasetSpec,
    materialize_tokenized_corpus,
    token_count_from_idx,
)


def ensure_token(token_env: str) -> str:
    token = os.getenv(token_env)
    if not token:
        raise RuntimeError(
            f"Environment variable {token_env} is not set. Export your Hugging Face token before running."
        )
    return token


def prepare_stage(
    stage_name: str,
    stage_cfg,
    data_cfg,
    hf_token: str,
    overwrite: bool,
) -> None:
    if stage_name not in data_cfg:
        raise RuntimeError(f"data.{stage_name} section missing from configuration")
    stage_data_cfg = data_cfg[stage_name]
    if "datasets" not in stage_data_cfg:
        raise RuntimeError(f"data.{stage_name}.datasets must be provided")

    dataset_specs = [DatasetSpec.from_config(entry) for entry in stage_data_cfg.datasets]
    if not dataset_specs:
        raise RuntimeError(f"data.{stage_name}.datasets is empty")

    output_prefix = Path(stage_data_cfg.get("output_prefix", stage_cfg.dataset_path)).expanduser()
    tokenizer_path = Path(stage_data_cfg.get("tokenizer_path", stage_cfg.tokenizer_path)).expanduser()

    target_tokens = stage_data_cfg.get("target_tokens")
    if target_tokens is not None:
        target_tokens = int(target_tokens)

    minimum_chars = int(stage_data_cfg.get("minimum_chars", 16))
    shuffle_buffer = int(stage_data_cfg.get("shuffle_buffer", 65536))

    stats = materialize_tokenized_corpus(
        tokenizer_path=tokenizer_path,
        specs=dataset_specs,
        output_prefix=output_prefix,
        hf_token=hf_token,
        target_tokens=target_tokens,
        minimum_chars=minimum_chars,
        shuffle_buffer=shuffle_buffer,
        overwrite=overwrite,
    )

    bin_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")
    dataset_path = Path(stage_cfg.dataset_path).expanduser()
    idx_target = Path(stage_cfg.idx_path).expanduser()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    idx_target.parent.mkdir(parents=True, exist_ok=True)

    if bin_path.resolve() != dataset_path.resolve():
        if dataset_path.exists():
            dataset_path.unlink()
        dataset_path.symlink_to(bin_path.resolve())
    if idx_path.resolve() != idx_target.resolve():
        if idx_target.exists():
            idx_target.unlink()
        idx_target.symlink_to(idx_path.resolve())

    tokens = token_count_from_idx(idx_path)
    print(
        f"✅ {stage_name} ready: {tokens:,} tokens across {stats['totals']['sequences']:,} sequences → {bin_path}"
    )


def materialize(args) -> None:
    config = OmegaConf.load(args.config)
    if "data" not in config:
        raise RuntimeError("configs/cloud_training.yaml is missing a top-level data section")

    stages: Iterable[str]
    if args.stage == "both":
        stages = ("stage1", "stage2")
    else:
        stages = (f"stage{args.stage}",)

    hf_token = ensure_token(config.data.get("hf_token_env", "HF_TOKEN"))

    for stage_name in stages:
        stage_cfg = getattr(config, stage_name)
        prepare_stage(stage_name, stage_cfg, config.data, hf_token, overwrite=args.overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cloud_training.yaml"),
        help="Configuration file defining dataset mixes",
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "both"],
        default="both",
        help="Which stage(s) to materialize",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild corpora even if existing tokenized shards meet the target",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize(args)


if __name__ == "__main__":
    main()
