#!/usr/bin/env python3
"""Minimal training test to isolate the exact issue."""

import time
import torch
import torch.nn.functional as F
from pathlib import Path

# Import our components
from fustercluck.data.tokenized_dataset import TokenizedDataset
from fustercluck.models.components import FusterCluckDecoder

def test_minimal_training():
    """Test training with just a few samples."""

    print("🔬 Testing minimal training setup...")

    # Load just the first few samples from dataset
    dataset_path = Path("data/tokenized/main.bin")
    idx_path = Path("data/tokenized/main.idx")

    print("Loading dataset...")
    dataset = TokenizedDataset(dataset_path, idx_path)
    print(f"Dataset has {len(dataset)} sequences")

    # Get just first 10 samples to avoid the packing issue
    samples = []
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        if len(sample) > 100:  # Skip very short samples
            samples.append(sample)
            if len(samples) >= 5:
                break

    print(f"Using {len(samples)} samples")
    for i, sample in enumerate(samples):
        print(f"Sample {i}: shape {sample.shape}")

    # Create simple batch (no packing complexity)
    max_len = max(len(s) for s in samples)
    batch_data = []
    for sample in samples:
        # Pad or truncate to max_len
        if len(sample) > max_len:
            batch_data.append(sample[:max_len])
        else:
            padded = torch.cat([sample, torch.zeros(max_len - len(sample), dtype=sample.dtype)])
            batch_data.append(padded)

    batch = torch.stack(batch_data)
    print(f"Created batch: shape {batch.shape}")

    # Create model
    print("Creating model...")
    vocab_size = 50000
    model = FusterCluckDecoder(
        vocab_size=vocab_size,
        dim=128,  # Much smaller for testing
        num_layers=2,  # Much smaller for testing
        num_heads=4,
        num_kv_heads=2,
        mlp_ratio=3.0,
        rope_theta=10000,
        dropout=0.0,
    ).to("cpu")

    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Starting minimal training loop...")

    # Minimal training loop - just one step
    model.train()
    step_start = time.time()

    # Get input and targets
    input_ids = batch[:, :-1]  # All but last token
    targets = batch[:, 1:]     # All but first token

    print(f"Input shape: {input_ids.shape}, Targets shape: {targets.shape}")

    # Forward pass
    print("Forward pass...")
    forward_start = time.time()
    logits = model(input_ids)
    forward_time = time.time() - forward_start
    print(".2f")

    # Loss computation
    print("Computing loss...")
    loss_start = time.time()
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss_time = time.time() - loss_start
    print(".2f")

    # Backward pass
    print("Backward pass...")
    backward_start = time.time()
    loss.backward()
    backward_time = time.time() - backward_start
    print(".2f")

    # Optimizer step
    print("Optimizer step...")
    opt_start = time.time()
    optimizer.step()
    optimizer.zero_grad()
    opt_time = time.time() - opt_start
    print(".2f")

    total_time = time.time() - step_start
    print(".2f")

    print("✅ Minimal training test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_minimal_training()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
