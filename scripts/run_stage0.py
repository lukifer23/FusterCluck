#!/usr/bin/env python3
"""CLI to run Stage 0 dry-run training."""

from __future__ import annotations

import argparse
from pathlib import Path

from fustercluck.train.config import OptimizerConfig, Stage0Config, TrainerConfig
from fustercluck.train.stage0 import run_stage0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_bin", type=Path, help="Path to tokenized dataset .bin file")
    parser.add_argument("data_idx", type=Path, help="Path to tokenized dataset .idx file")
    parser.add_argument("tokenizer", type=Path, help="SentencePiece model path")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints/stage0"))
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_cfg = Stage0Config(
        dataset_path=args.data_bin,
        idx_path=args.data_idx,
        tokenizer_path=args.tokenizer,
        max_steps=args.max_steps,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch,
        gradient_accumulation=args.grad_accum,
        precision=args.precision,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_dir=args.checkpoint_dir,
        optimizer=OptimizerConfig(lr=args.lr),
    )
    trainer_cfg = TrainerConfig(
        device=args.device,
        grad_clip=1.0,
        use_compile=args.compile,
        precision=args.precision,
    )
    run_stage0(stage_cfg, trainer_cfg)


if __name__ == "__main__":
    main()
