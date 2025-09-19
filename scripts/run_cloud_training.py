#!/usr/bin/env python3
"""Main cloud training script for FusterCluck."""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from omegaconf import OmegaConf

from fustercluck.train.stage0 import Stage0Trainer
from fustercluck.train.config import Stage0Config, TrainerConfig
from fustercluck.data.tokenized_dataset import TokenizedDataset
from fustercluck.utils.checkpoint import CheckpointManager

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
        os.system("python scripts/download_data.py --datasets refinedweb science --output-dir data/raw/cloud")
        
        # Process data
        os.system("python scripts/process_cloud_data.py --input-dir data/raw/cloud --output-dir data/processed/cloud")
        
        # Build tokenizer if not exists
        if not Path("artifacts/tokenizer/fustercluck.model").exists():
            logger.info("Building tokenizer...")
            os.system("python scripts/build_tokenizer.py data/processed/cloud --output artifacts/tokenizer/fustercluck --vocab-size 50000")
        
        # Tokenize data
        logger.info("Tokenizing data...")
        os.system(f"python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage1 data/processed/cloud/refinedweb_processed.txt")
        os.system(f"python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage2 data/processed/cloud/science_processed.txt")
        
    def run_stage1(self):
        """Run Stage 1 training (2B tokens)."""
        logger.info("Starting Stage 1 training...")
        
        # Convert config to dataclass
        stage1_cfg = Stage0Config(
            dataset_path=self.stage1_config.dataset_path,
            idx_path=self.stage1_config.idx_path,
            tokenizer_path=self.stage1_config.tokenizer_path,
            max_steps=self.stage1_config.max_steps,
            seq_len=self.stage1_config.seq_len,
            micro_batch_size=self.stage1_config.micro_batch_size,
            gradient_accumulation=self.stage1_config.gradient_accumulation,
            precision=self.stage1_config.precision,
            log_interval=self.stage1_config.log_interval,
            eval_interval=self.stage1_config.eval_interval,
            checkpoint_dir=self.stage1_config.checkpoint_dir,
            optimizer=type('obj', (object,), self.stage1_config.optimizer)()
        )
        
        trainer_cfg = TrainerConfig(
            device=self.trainer_config.device,
            grad_clip=self.stage1_config.grad_clip,
            use_compile=self.trainer_config.use_compile,
            compile_mode=self.trainer_config.compile_mode,
            precision=self.trainer_config.precision
        )
        
        # Run training
        trainer = Stage0Trainer(stage1_cfg, trainer_cfg)
        trainer.train()
        
        logger.info("Stage 1 training complete!")
        
    def run_stage2(self):
        """Run Stage 2 training (5B tokens)."""
        logger.info("Starting Stage 2 training...")
        
        # Similar to stage1 but with stage2 config
        stage2_cfg = Stage0Config(
            dataset_path=self.stage2_config.dataset_path,
            idx_path=self.stage2_config.idx_path,
            tokenizer_path=self.stage2_config.tokenizer_path,
            max_steps=self.stage2_config.max_steps,
            seq_len=self.stage2_config.seq_len,
            micro_batch_size=self.stage2_config.micro_batch_size,
            gradient_accumulation=self.stage2_config.gradient_accumulation,
            precision=self.stage2_config.precision,
            log_interval=self.stage2_config.log_interval,
            eval_interval=self.stage2_config.eval_interval,
            checkpoint_dir=self.stage2_config.checkpoint_dir,
            optimizer=type('obj', (object,), self.stage2_config.optimizer)()
        )
        
        trainer_cfg = TrainerConfig(
            device=self.trainer_config.device,
            grad_clip=self.stage2_config.grad_clip,
            use_compile=self.trainer_config.use_compile,
            compile_mode=self.trainer_config.compile_mode,
            precision=self.trainer_config.precision
        )
        
        # Run training
        trainer = Stage0Trainer(stage2_cfg, trainer_cfg)
        trainer.train()
        
        logger.info("Stage 2 training complete!")
        
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
    parser.add_argument("--stage", type=str, choices=["1", "2", "both"], default="both")
    parser.add_argument("--skip-data", action="store_true", help="Skip data download/processing")
    
    args = parser.parse_args()
    
    trainer = CloudTrainer(args.config)
    
    if not args.skip_data:
        trainer.download_and_process_data()
    
    if args.stage in ["1", "both"]:
        trainer.run_stage1()
    
    if args.stage in ["2", "both"]:
        trainer.run_stage2()

if __name__ == "__main__":
    main()
