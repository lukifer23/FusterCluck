#!/usr/bin/env python3
"""Sequentially execute text training stages defined in a YAML config."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable

from omegaconf import OmegaConf

from fustercluck.train.config import (
    CheckpointConfig,
    OptimizerConfig,
    StageConfig,
    TrainerConfig,
)
from fustercluck.train.stage0 import TextStageTrainer


def _to_path(path_value: str | Path) -> Path:
    return path_value if isinstance(path_value, Path) else Path(path_value)


def _optimizer_kwargs(raw: Dict[str, Any] | None) -> Dict[str, Any]:
    return dict(raw or {})


def _stage_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(raw)
    data.pop("name", None)
    data.pop("trainer", None)
    data["dataset_path"] = _to_path(data["dataset_path"])
    data["idx_path"] = _to_path(data["idx_path"])
    data["tokenizer_path"] = _to_path(data["tokenizer_path"])
    if "checkpoint_dir" in data:
        data["checkpoint_dir"] = _to_path(data["checkpoint_dir"])
    if "optimizer" in data:
        data["optimizer"] = OptimizerConfig(**_optimizer_kwargs(data["optimizer"]))
    else:
        data["optimizer"] = OptimizerConfig()
    return data


def _trainer_kwargs(raw: Dict[str, Any] | None) -> Dict[str, Any]:
    data = dict(raw or {})
    if "checkpoint" in data and data["checkpoint"] is not None:
        ckpt = dict(data["checkpoint"])
        ckpt["directory"] = _to_path(ckpt["directory"])
        ckpt.setdefault("keep_last", 3)
        data["checkpoint"] = CheckpointConfig(**ckpt)
    return data


def run_pipeline(config_path: Path, stages: Iterable[str] | None = None) -> None:
    cfg = OmegaConf.load(config_path)
    stage_entries = OmegaConf.to_container(cfg.get("stages", []), resolve=True)
    if not isinstance(stage_entries, list) or not stage_entries:
        raise RuntimeError("Config must contain a non-empty 'stages' list")

    trainer_defaults = TrainerConfig(**_trainer_kwargs(OmegaConf.to_container(cfg.get("trainer", {}), resolve=True)))

    requested = set(stages) if stages else None

    for stage_cfg in stage_entries:
        name = stage_cfg.get("name")
        if not name:
            raise RuntimeError("Each stage entry requires a 'name'")
        if requested and name not in requested:
            continue
        print(f"=== Running stage: {name} ===")
        trainer_cfg = trainer_defaults
        if "trainer" in stage_cfg:
            overrides = _trainer_kwargs(stage_cfg["trainer"])
            trainer_cfg = replace(trainer_defaults, **overrides)
        stage_kwargs = _stage_kwargs(stage_cfg)
        if "checkpoint_dir" in stage_kwargs:
            base_ckpt_dir = stage_kwargs.pop("checkpoint_dir")
        elif trainer_cfg.checkpoint is not None:
            base_ckpt_dir = trainer_cfg.checkpoint.directory
        else:
            base_ckpt_dir = Path("artifacts/checkpoints")
        checkpoint_dir = _to_path(base_ckpt_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stage_kwargs["checkpoint_dir"] = checkpoint_dir
        stage = StageConfig(**stage_kwargs)
        trainer = TextStageTrainer(stage, trainer_cfg)
        trainer.train()
        print(f"=== Stage {name} finished ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML config describing the stages")
    parser.add_argument(
        "--stages",
        nargs="*",
        help="Optional subset of stage names to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, stages=args.stages)


if __name__ == "__main__":
    main()
