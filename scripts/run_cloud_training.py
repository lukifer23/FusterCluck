#!/usr/bin/env python3
"""Main cloud training script for FusterCluck."""

import argparse
import logging
import os
import time
import glob
import re
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf

from fustercluck.train.stage0 import Stage0Trainer
from fustercluck.train.config import (
    OptimizerConfig,
    Stage0Config,
    StageVisionConfig,
    TrainerConfig,
    VisionAdapterConfig,
)
from fustercluck.train.stage_multimodal import StageMultimodalTrainer
from fustercluck.data.corpus_builder import (
    DatasetSpec,
    materialize_tokenized_corpus,
    token_count_from_idx,
)

# Set up logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'cloud_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CloudTrainer:
    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)
        self.cloud_config = self.config.cloud
        self.stage1_config = self.config.stage1
        self.stage2_config = self.config.stage2
        self.trainer_config = self.config.trainer
        self.data_config = getattr(self.config, "data", None)
        
        # Set up directories
        self.setup_directories()
        
        # Set up monitoring
        self.setup_monitoring()

    def _build_stage_config(self, cfg) -> Stage0Config:
        """Convert OmegaConf DictConfig into Stage0Config with proper Path objects."""
        optimizer_cfg = OptimizerConfig(
            **OmegaConf.to_container(cfg.optimizer, resolve=True)
        )
        return Stage0Config(
            dataset_path=Path(cfg.dataset_path),
            idx_path=Path(cfg.idx_path),
            tokenizer_path=Path(cfg.tokenizer_path),
            max_steps=cfg.max_steps,
            seq_len=cfg.seq_len,
            micro_batch_size=cfg.micro_batch_size,
            gradient_accumulation=cfg.gradient_accumulation,
            precision=cfg.precision,
            log_interval=cfg.log_interval,
            eval_interval=cfg.eval_interval,
            checkpoint_dir=Path(cfg.checkpoint_dir),
            model_dim=getattr(cfg, "model_dim", 1024),
            model_layers=getattr(cfg, "model_layers", 4),
            model_heads=getattr(cfg, "model_heads", 16),
            model_kv_heads=getattr(cfg, "model_kv_heads", 4),
            mlp_ratio=getattr(cfg, "mlp_ratio", 4.0),
            rope_theta=getattr(cfg, "rope_theta", 10000),
            dropout=getattr(cfg, "dropout", 0.0),
            optimizer=optimizer_cfg,
        )

    def _expand_braces(self, pattern: str) -> list[str]:
        brace_re = re.compile(r"\{(\d+)\.\.(\d+)\}")

        match = brace_re.search(pattern)
        if not match:
            return [pattern]
        start, end = int(match.group(1)), int(match.group(2))
        if end < start:
            start, end = end, start
        width = max(len(match.group(1)), len(match.group(2)))
        expanded: list[str] = []
        for value in range(start, end + 1):
            replacement = f"{value:0{width}d}"
            replaced = pattern[: match.start()] + replacement + pattern[match.end():]
            expanded.extend(self._expand_braces(replaced))
        return expanded

    def _resolve_vision_shards(self, stage: str, patterns: Sequence[str]) -> list[str]:
        resolved: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            expanded_env = os.path.expanduser(os.path.expandvars(pattern))
            candidates = self._expand_braces(expanded_env)
            matched = False
            for candidate in candidates:
                matches = sorted(glob.glob(candidate))
                if not matches:
                    continue
                matched = True
                for match_path in matches:
                    resolved_path = str(Path(match_path).resolve())
                    if resolved_path not in seen:
                        seen.add(resolved_path)
                        resolved.append(resolved_path)
            if not matched:
                logger.warning("%s: pattern %s produced no matches", stage, expanded_env)

        if not resolved:
            raise RuntimeError(f"No vision shards found for {stage}. Patterns={list(patterns)}")

        logger.info("%s resolved to %d vision shards", stage, len(resolved))
        return resolved

    def _build_vision_config(self, cfg, stage: str) -> StageVisionConfig:
        params = OmegaConf.to_container(cfg, resolve=True)
        optimizer_cfg = OptimizerConfig(**params.get("optimizer", {}))
        adapter_cfg = VisionAdapterConfig(**params.get("adapter", {}))
        shards = params.get("vision_shards", [])
        resolved_shards = self._resolve_vision_shards(stage, shards)
        return StageVisionConfig(
            tokenizer_path=Path(params["tokenizer_path"]),
            max_steps=int(params.get("max_steps", 1000)),
            seq_len=int(params["seq_len"]),
            micro_batch_size=int(params["micro_batch_size"]),
            gradient_accumulation=int(params["gradient_accumulation"]),
            precision=params.get("precision", "bf16"),
            log_interval=int(params.get("log_interval", 100)),
            eval_interval=int(params.get("eval_interval", 500)),
            checkpoint_dir=Path(params["checkpoint_dir"]),
            model_dim=int(params.get("model_dim", 1024)),
            model_layers=int(params.get("model_layers", 24)),
            model_heads=int(params.get("model_heads", 16)),
            model_kv_heads=int(params.get("model_kv_heads", 4)),
            mlp_ratio=float(params.get("mlp_ratio", 4.0)),
            rope_theta=int(params.get("rope_theta", 10000)),
            dropout=float(params.get("dropout", 0.0)),
            optimizer=optimizer_cfg,
            vision_shards=resolved_shards,
            shuffle_buffer=int(params.get("shuffle_buffer", 2048)),
            image_token=params.get("image_token", "<image>"),
            image_token_id=params.get("image_token_id"),
            adapter=adapter_cfg,
        )

    def setup_directories(self):
        """Create necessary directories."""
        dirs = [
            "data/raw/cloud",
            "data/processed/cloud", 
            "data/tokenized/cloud",
            "artifacts/checkpoints/cloud/stage1",
            "artifacts/checkpoints/cloud/stage2",
            "logs"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def setup_monitoring(self):
        """Set up monitoring tools."""
        if self.config.monitoring.wandb:
            import wandb
            wandb.init(
                project=self.config.monitoring.wandb.project,
                entity=self.config.monitoring.wandb.entity,
                tags=self.config.monitoring.wandb.tags,
                config=OmegaConf.to_container(self.config, resolve=True)
            )
            
    def _prepare_stage_text(self, stage: str, cfg) -> None:
        if not self.data_config or stage not in self.data_config:
            raise RuntimeError(f"data.{stage} configuration missing from YAML")

        stage_data_cfg = self.data_config[stage]
        if "datasets" not in stage_data_cfg:
            raise RuntimeError(f"data.{stage}.datasets must be provided")

        dataset_specs = [
            DatasetSpec.from_config(item) for item in stage_data_cfg.get("datasets")
        ]

        if stage == "stage1":
            dataset_path = Path(cfg.dataset_path)
            idx_path = Path(cfg.idx_path)
        elif stage == "stage2":
            dataset_path = Path(cfg.dataset_path)
            idx_path = Path(cfg.idx_path)
        else:
            raise ValueError(f"Unsupported text stage: {stage}")

        output_prefix = Path(stage_data_cfg.get("output_prefix", dataset_path.with_suffix("")))
        tokenizer_path = Path(stage_data_cfg.get("tokenizer_path", cfg.tokenizer_path))

        target_tokens = stage_data_cfg.get("target_tokens")
        if target_tokens is not None:
            target_tokens = int(target_tokens)
        minimum_chars = int(stage_data_cfg.get("minimum_chars", 16))
        shuffle_buffer = int(stage_data_cfg.get("shuffle_buffer", 8192))
        overwrite = bool(stage_data_cfg.get("overwrite", False))

        hf_token_env = stage_data_cfg.get(
            "hf_token_env", self.data_config.get("hf_token_env", "HF_TOKEN")
        )
        hf_token = os.getenv(hf_token_env)
        if not hf_token:
            raise RuntimeError(
                f"Environment variable {hf_token_env} is not set; required for {stage} data"
            )

        bin_path = output_prefix.with_suffix(".bin")
        idx_out_path = output_prefix.with_suffix(".idx")

        if not overwrite and bin_path.exists() and idx_out_path.exists():
            try:
                existing_tokens = token_count_from_idx(idx_out_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to inspect existing idx file %s: %s", idx_out_path, exc)
                existing_tokens = 0
            if target_tokens is None or existing_tokens >= target_tokens:
                logger.info(
                    "%s corpus already materialized (%d tokens) – reusing %s",
                    stage,
                    existing_tokens,
                    bin_path,
                )
                return
            logger.info(
                "%s corpus has %d tokens (<%s target). Rebuilding with overwrite.",
                stage,
                existing_tokens,
                target_tokens,
            )
            overwrite = True

        logger.info(
            "Materializing %s corpus to %s (target_tokens=%s, datasets=%d)",
            stage,
            output_prefix,
            target_tokens,
            len(dataset_specs),
        )

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

        logger.info(
            "%s corpus ready (%d tokens, %d sequences)",
            stage,
            stats["totals"]["tokens"],
            stats["totals"]["sequences"],
        )

        if bin_path != dataset_path:
            logger.debug("Linking %s -> %s", bin_path, dataset_path)
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            if dataset_path.exists():
                dataset_path.unlink()
            dataset_path.symlink_to(bin_path.resolve())
        if idx_out_path != idx_path:
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            if idx_path.exists():
                idx_path.unlink()
            idx_path.symlink_to(idx_out_path.resolve())

    def download_and_process_data(self):
        """Materialize Stage 1/2 tokenized corpora using real datasets."""
        logger.info("Preparing text corpora for cloud training...")
        self._prepare_stage_text("stage1", self.stage1_config)
        self._prepare_stage_text("stage2", self.stage2_config)

    def run_stage1(self, resume: bool = False):
        """Run Stage 1 training (2B tokens)."""
        logger.info("Starting Stage 1 training...")

        # Convert config to dataclass with proper Path/optimizer types
        stage1_cfg = self._build_stage_config(self.stage1_config)

        trainer_cfg = TrainerConfig(
            device=self.trainer_config.device,
            grad_clip=self.stage1_config.grad_clip,
            use_compile=self.trainer_config.use_compile,
            compile_mode=self.trainer_config.compile_mode,
            precision=self.trainer_config.precision,
            dataloader_workers=getattr(self.trainer_config, "dataloader_workers", 0),
            pin_memory=getattr(self.trainer_config, "pin_memory", False),
            persistent_workers=getattr(self.trainer_config, "persistent_workers", False),
            env=OmegaConf.to_container(getattr(self.trainer_config, "env", {}), resolve=True)
            if getattr(self.trainer_config, "env", None)
            else None,
        )
        if getattr(self.trainer_config, "env", None):
            os.environ.update({k: str(v) for k, v in self.trainer_config.env.items()})

        # Run training
        trainer = Stage0Trainer(stage1_cfg, trainer_cfg)
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        logger.info("Stage1 model params: total=%d trainable=%d", total_params, trainable_params)
        if resume:
            latest = trainer.checkpoint.latest()
            if latest:
                logger.info("Resuming Stage 1 from %s", latest)
                trainer.resume_from_checkpoint(latest)
            else:
                logger.info("No Stage 1 checkpoint found; starting from scratch")
        trainer.train()

        logger.info("Stage 1 training complete!")

    def run_stage2(self, resume: bool = False):
        """Run Stage 2 training (5B tokens)."""
        logger.info("Starting Stage 2 training...")

        # Similar to stage1 but with stage2 config
        stage2_cfg = self._build_stage_config(self.stage2_config)

        trainer_cfg = TrainerConfig(
            device=self.trainer_config.device,
            grad_clip=self.stage2_config.grad_clip,
            use_compile=self.trainer_config.use_compile,
            compile_mode=self.trainer_config.compile_mode,
            precision=self.trainer_config.precision,
            dataloader_workers=getattr(self.trainer_config, "dataloader_workers", 0),
            pin_memory=getattr(self.trainer_config, "pin_memory", False),
            persistent_workers=getattr(self.trainer_config, "persistent_workers", False),
            env=OmegaConf.to_container(getattr(self.trainer_config, "env", {}), resolve=True)
            if getattr(self.trainer_config, "env", None)
            else None,
        )
        if getattr(self.trainer_config, "env", None):
            os.environ.update({k: str(v) for k, v in self.trainer_config.env.items()})

        # Run training
        trainer = Stage0Trainer(stage2_cfg, trainer_cfg)
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        logger.info("Stage2 model params: total=%d trainable=%d", total_params, trainable_params)
        if resume:
            latest = trainer.checkpoint.latest()
            if latest:
                logger.info("Resuming Stage 2 from %s", latest)
                trainer.resume_from_checkpoint(latest)
            else:
                logger.info("No Stage 2 checkpoint found; starting from scratch")
        trainer.train()

        logger.info("Stage 2 training complete!")

    def run_stage3(self, resume: bool = False):
        """Run Stage 3 multimodal alignment."""
        if "stage3" not in self.config:
            raise RuntimeError("stage3 configuration missing from YAML")
        vision_cfg = self._build_vision_config(self.config.stage3, "stage3")
        trainer_cfg = TrainerConfig(
            device=self.trainer_config.device,
            grad_clip=vision_cfg.grad_clip,
            use_compile=self.trainer_config.use_compile,
            compile_mode=self.trainer_config.compile_mode,
            precision=self.trainer_config.precision,
            dataloader_workers=getattr(self.trainer_config, "dataloader_workers", 0),
            pin_memory=getattr(self.trainer_config, "pin_memory", False),
            persistent_workers=getattr(self.trainer_config, "persistent_workers", False),
            env=OmegaConf.to_container(getattr(self.trainer_config, "env", {}), resolve=True)
            if getattr(self.trainer_config, "env", None)
            else None,
        )
        if getattr(self.trainer_config, "env", None):
            os.environ.update({k: str(v) for k, v in self.trainer_config.env.items()})
        trainer = StageMultimodalTrainer(vision_cfg, trainer_cfg)
        total_params = sum(p.numel() for p in trainer.model.parameters()) + sum(
            p.numel() for p in trainer.adapter.parameters()
        )
        trainable_params = sum(
            p.numel() for p in list(trainer.model.parameters()) + list(trainer.adapter.parameters()) if p.requires_grad
        )
        logger.info("Stage3 model params: total=%d trainable=%d", total_params, trainable_params)
        if resume:
            latest = trainer.checkpoint.latest()
            if latest:
                trainer.resume_from_checkpoint(latest)
            else:
                logger.info("No Stage 3 checkpoint found; starting from scratch")
        trainer.train()
        logger.info("Stage 3 training complete!")

    def run_stage4(self, resume: bool = False):
        """Run Stage 4 multimodal training."""
        if "stage4" not in self.config:
            raise RuntimeError("stage4 configuration missing from YAML")
        vision_cfg = self._build_vision_config(self.config.stage4, "stage4")
        trainer_cfg = TrainerConfig(
            device=self.trainer_config.device,
            grad_clip=self.stage1_config.grad_clip,
            use_compile=self.trainer_config.use_compile,
            compile_mode=self.trainer_config.compile_mode,
            precision=self.trainer_config.precision,
            dataloader_workers=getattr(self.trainer_config, "dataloader_workers", 0),
            pin_memory=getattr(self.trainer_config, "pin_memory", False),
            persistent_workers=getattr(self.trainer_config, "persistent_workers", False),
            env=OmegaConf.to_container(getattr(self.trainer_config, "env", {}), resolve=True)
            if getattr(self.trainer_config, "env", None)
            else None,
        )
        if getattr(self.trainer_config, "env", None):
            os.environ.update({k: str(v) for k, v in self.trainer_config.env.items()})
        trainer = StageMultimodalTrainer(vision_cfg, trainer_cfg)
        total_params = sum(p.numel() for p in trainer.model.parameters()) + sum(
            p.numel() for p in trainer.adapter.parameters()
        )
        trainable_params = sum(
            p.numel() for p in list(trainer.model.parameters()) + list(trainer.adapter.parameters()) if p.requires_grad
        )
        logger.info("Stage4 model params: total=%d trainable=%d", total_params, trainable_params)
        if resume:
            latest = trainer.checkpoint.latest()
            if latest:
                trainer.resume_from_checkpoint(latest)
            else:
                logger.info("No Stage 4 checkpoint found; starting from scratch")
        trainer.train()
        logger.info("Stage 4 training complete!")
    def run_full_training(self):
        """Run complete cloud training pipeline."""
        logger.info("Starting full cloud training pipeline...")
        
        start_time = time.time()
        
        # Download and process data
        self.download_and_process_data()
        
        # Run Stage 1
        self.run_stage1()
        
        # Run Stage 2
        self.run_stage2()
        
        total_time = time.time() - start_time
        logger.info(f"Full training pipeline complete in {total_time/3600:.2f} hours")
        
        # Upload final checkpoints
        self.upload_checkpoints()
        
    def upload_checkpoints(self):
        """Upload final checkpoints to cloud storage."""
        logger.info("Uploading checkpoints...")
        
        # This would integrate with your preferred cloud storage
        # For now, just log the checkpoint locations
        checkpoint_dirs = [
            "artifacts/checkpoints/cloud/stage1",
            "artifacts/checkpoints/cloud/stage2"
        ]
        
        for checkpoint_dir in checkpoint_dirs:
            if Path(checkpoint_dir).exists():
                logger.info(f"Checkpoints available in: {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run FusterCluck cloud training")
    parser.add_argument("--config", type=str, default="configs/cloud_training.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "3", "4", "both", "vision", "all", "text"],
        default="both",
    )
    parser.add_argument("--skip-data", action="store_true", help="Skip data download/processing")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoints for selected stages")
    
    args = parser.parse_args()
    
    trainer = CloudTrainer(args.config)

    textual_stages = {"1", "2", "both", "all", "text"}
    if args.stage in textual_stages and not args.skip_data:
        trainer.download_and_process_data()

    if args.stage in {"1", "both", "all", "text"}:
        trainer.run_stage1(resume=args.resume)

    if args.stage in {"2", "both", "all", "text"}:
        trainer.run_stage2(resume=args.resume)

    if args.stage in {"3", "vision", "all"}:
        trainer.run_stage3(resume=args.resume)

    if args.stage in {"4", "vision", "all"}:
        trainer.run_stage4(resume=args.resume)

if __name__ == "__main__":
    main()
