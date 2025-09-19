#!/usr/bin/env python3
"""Main cloud training script for FusterCluck."""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Iterable

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/cloud_training.log'),
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

    def _build_vision_config(self, cfg) -> StageVisionConfig:
        params = OmegaConf.to_container(cfg, resolve=True)
        optimizer_cfg = OptimizerConfig(**params.get("optimizer", {}))
        adapter_cfg = VisionAdapterConfig(**params.get("adapter", {}))
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
            vision_shards=params.get("vision_shards", []),
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
            
    def download_and_process_data(self):
        """Download and process training data."""
        logger.info("Downloading and processing data...")
        
        # Download data
        os.system(
            "python scripts/download_data.py --domains science data code chess general --output-dir data/raw/cloud"
        )
        
        # Process data
        os.system(
            "python scripts/process_cloud_data.py --input-dir data/raw/cloud --output-dir data/processed/cloud"
        )

        processed_dir = Path("data/processed/cloud")
        self._build_stage_shards(processed_dir)
        
        # Build tokenizer if not exists
        if not Path("artifacts/tokenizer/fustercluck.model").exists():
            logger.info("Building tokenizer...")
            os.system("python scripts/build_tokenizer.py data/processed/cloud --output artifacts/tokenizer/fustercluck --vocab-size 50000")
        
        # Tokenize data
        logger.info("Tokenizing data...")
        os.system(
            "python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model "
            "data/tokenized/cloud/stage1 data/processed/cloud/stage1_mix.txt"
        )
        os.system(
            "python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model "
            "data/tokenized/cloud/stage2 data/processed/cloud/stage2_mix.txt"
        )

    def _build_stage_shards(self, processed_dir: Path) -> None:
        """Combine processed domain files into stage-level text shards."""

        def concat(domains: Iterable[str], output_name: str) -> None:
            output_path = processed_dir / output_name
            with output_path.open("w", encoding="utf-8") as handle:
                for domain in domains:
                    domain_path = processed_dir / f"{domain}_processed.txt"
                    if not domain_path.exists():
                        logger.warning("Missing processed slice %s", domain_path)
                        continue
                    handle.write(domain_path.read_text(encoding="utf-8"))
                    handle.write("\n")
            logger.info("Wrote %s", output_path)

        concat(["science", "data", "code"], "stage1_mix.txt")
        concat(["science", "data", "code", "chess", "general"], "stage2_mix.txt")
        
    def run_stage1(self, resume: bool = False):
        """Run Stage 1 training (2B tokens)."""
        logger.info("Starting Stage 1 training...")

        # Convert config to dataclass with proper Path/optimizer types
        stage1_cfg = self._build_stage_config(self.stage1_config)

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
        vision_cfg = self._build_vision_config(self.config.stage3)
        if not vision_cfg.vision_shards:
            raise RuntimeError("stage3.vision_shards is empty – provide WebDataset shards")
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
        vision_cfg = self._build_vision_config(self.config.stage4)
        if not vision_cfg.vision_shards:
            raise RuntimeError("stage4.vision_shards is empty – provide WebDataset shards")
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
