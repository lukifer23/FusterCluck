#!/usr/bin/env python3
"""CLI entrypoint for running a single text training stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from fustercluck.train.config import OptimizerConfig, StageConfig, TrainerConfig
from fustercluck.train.stage0 import run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_bin", type=Path, help="Path to tokenized dataset .bin file")
    parser.add_argument("data_idx", type=Path, help="Path to tokenized dataset .idx file")
    parser.add_argument("tokenizer", type=Path, help="SentencePiece model path")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=12288)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--model-dim", type=int, default=896)
    parser.add_argument("--layers", type=int, default=18)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=3.5)
    parser.add_argument("--rope-theta", type=int, default=10000)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints/text-stage"))
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_cfg = StageConfig(
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
        model_dim=args.model_dim,
        model_layers=args.layers,
        model_heads=args.heads,
        model_kv_heads=args.kv_heads,
        mlp_ratio=args.mlp_ratio,
        rope_theta=args.rope_theta,
        dropout=args.dropout,
        optimizer=OptimizerConfig(lr=args.lr),
    )
    trainer_cfg = TrainerConfig(
        device=args.device,
        grad_clip=1.0,
        use_compile=args.compile,
        precision=args.precision,
        gradient_checkpointing=args.grad_checkpointing,
    )
    run_stage(stage_cfg, trainer_cfg)


if __name__ == "__main__":
    main()
