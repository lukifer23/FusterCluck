#!/usr/bin/env python3
"""Debug script to isolate training issues."""

import time
import torch
from pathlib import Path

# Test imports
print("Testing imports...")
from fustercluck.data.tokenized_dataset import TokenizedDataset
from fustercluck.models.components import FusterCluckDecoder
from fustercluck.train.stage0 import PackedSequenceDataset, create_dataloader
from fustercluck.train.config import StageConfig, TrainerConfig

print("✅ Imports successful")

# Test dataset loading
print("Testing dataset loading...")
dataset_path = Path("data/tokenized/main.bin")
idx_path = Path("data/tokenized/main.idx")

try:
    start_time = time.time()
    dataset = TokenizedDataset(dataset_path, idx_path)
    load_time = time.time() - start_time
    print(f"✅ Dataset loaded in {load_time:.2f}s, length: {len(dataset)}")
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    exit(1)

# Test getting one sample
print("Testing sample retrieval...")
try:
    start_time = time.time()
    sample = dataset[0]
    sample_time = time.time() - start_time
    print(f"✅ Sample retrieved in {sample_time:.2f}s, shape: {sample.shape}")
except Exception as e:
    print(f"❌ Sample retrieval failed: {e}")
    exit(1)

# Test dataloader creation
print("Testing dataloader creation...")
try:
    cfg = StageConfig(
        dataset_path=dataset_path,
        idx_path=idx_path,
        tokenizer_path=Path("artifacts/tokenizer/fustercluck.model"),
        seq_len=8192,
        micro_batch_size=1,
        gradient_accumulation=16,
        max_steps=10
    )
    trainer_cfg = TrainerConfig(device="cpu")

    start_time = time.time()
    dataloader = create_dataloader(dataset, cfg, trainer_cfg)
    dataloader_time = time.time() - start_time
    print(f"✅ Dataloader created in {dataloader_time:.2f}s")
except Exception as e:
    print(f"❌ Dataloader creation failed: {e}")
    exit(1)

# Test getting one batch
print("Testing batch retrieval...")
try:
    start_time = time.time()
    batch = next(iter(dataloader))
    batch_time = time.time() - start_time
    print(f"✅ Batch retrieved in {batch_time:.2f}s, shape: {batch.shape}")
except Exception as e:
    print(f"❌ Batch retrieval failed: {e}")
    exit(1)

# Test model creation
print("Testing model creation...")
try:
    vocab_size = 50000  # Approximate
    start_time = time.time()
    model = FusterCluckDecoder(
        vocab_size=vocab_size,
        dim=896,
        num_layers=18,
        num_heads=14,
        num_kv_heads=2,
        mlp_ratio=3.5,
        rope_theta=10000,
        dropout=0.0,
    )
    model_time = time.time() - start_time
    print(f"✅ Model created in {model_time:.2f}s")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    exit(1)

# Test model forward pass
print("Testing model forward pass...")
try:
    model = model.to("cpu")
    input_ids = batch[:, :-1]  # Remove last token for targets
    start_time = time.time()
    with torch.no_grad():
        logits = model(input_ids)
    forward_time = time.time() - start_time
    print(f"✅ Forward pass completed in {forward_time:.2f}s, output shape: {logits.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    exit(1)

print("🎉 All tests passed! Training pipeline components are working.")
