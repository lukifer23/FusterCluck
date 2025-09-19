#!/usr/bin/env python3
"""Scale up to actually reach 2B tokens based on real tokenization results."""

import os
import sys
from pathlib import Path
import random

def estimate_tokens_from_samples(samples: int, avg_tokens_per_sample: int = 250) -> int:
    """Estimate total tokens from sample count."""
    return samples * avg_tokens_per_sample

def scale_up_to_target():
    """Scale up data to actually reach 2B tokens."""
    print("🎯 Scaling up to actually reach 2B tokens...")
    
    # Based on actual results: 9.3M samples = 193M tokens
    # So we need: 2B tokens / 193M tokens = ~10.4x more data
    
    stage1_file = Path("data/processed/cloud/stage1_mix.txt")
    stage2_file = Path("data/processed/cloud/stage2_mix.txt")
    
    if not stage1_file.exists():
        print("❌ Stage 1 file not found!")
        return False
    
    # Read original data
    with open(stage1_file, 'r') as f:
        original_lines = [line.strip() for line in f if line.strip()]
    
    original_count = len(original_lines)
    print(f"📊 Original Stage 1: {original_count:,} samples")
    
    # Calculate expansion needed for 2B tokens
    # Current: 193M tokens from 9.3M samples = ~20.7 tokens per sample
    tokens_per_sample = 193_313_804 / 9_318_458  # ~20.7 tokens per sample
    
    target_samples = 2_000_000_000 / tokens_per_sample  # ~96.6M samples needed
    expansion_factor = int(target_samples / original_count) + 1
    
    print(f"📊 Current tokens per sample: {tokens_per_sample:.1f}")
    print(f"📊 Target samples for 2B tokens: {target_samples:,.0f}")
    print(f"📊 Expansion factor needed: {expansion_factor}x")
    
    # Create expanded Stage 1 with progress tracking
    print("🔄 Creating expanded Stage 1...")
    expanded_stage1 = []
    
    # Show progress for expansion
    from tqdm import tqdm
    for i in tqdm(range(expansion_factor), desc="📈 Expanding Stage 1", unit="cycles"):
        expanded_stage1.extend(original_lines)
        if (i + 1) % 5 == 0:  # Update every 5 cycles
            print(f"  📊 Progress: {len(expanded_stage1):,} samples, ~{len(expanded_stage1) * tokens_per_sample:,.0f} tokens")
    
    # Add random samples to reach exact target
    remaining_needed = int(target_samples) - len(expanded_stage1)
    if remaining_needed > 0:
        print(f"🔄 Adding {remaining_needed:,} random samples to reach target...")
        expanded_stage1.extend(random.choices(original_lines, k=remaining_needed))
    
    # Shuffle and save with progress
    print("🔄 Shuffling and saving Stage 1...")
    random.shuffle(expanded_stage1)
    
    print("💾 Writing Stage 1 file...")
    with open(stage1_file, 'w', encoding='utf-8') as f:
        # Write in chunks to show progress
        chunk_size = 100000  # 100K lines per chunk
        for i in tqdm(range(0, len(expanded_stage1), chunk_size), desc="💾 Writing Stage 1", unit="chunks"):
            chunk = expanded_stage1[i:i+chunk_size]
            f.write('\n'.join(chunk) + '\n')
    
    print(f"✅ Stage 1 expanded to {len(expanded_stage1):,} samples")
    print(f"📊 Estimated tokens: {len(expanded_stage1) * tokens_per_sample:,.0f}")
    
    # Do the same for Stage 2 (5B tokens)
    if stage2_file.exists():
        with open(stage2_file, 'r') as f:
            original_stage2_lines = [line.strip() for line in f if line.strip()]
        
        original_stage2_count = len(original_stage2_lines)
        print(f"📊 Original Stage 2: {original_stage2_count:,} samples")
        
        # Calculate expansion for 5B tokens
        target_stage2_samples = 5_000_000_000 / tokens_per_sample  # ~241.5M samples needed
        stage2_expansion_factor = int(target_stage2_samples / original_stage2_count) + 1
        
        print(f"📊 Target Stage 2 samples for 5B tokens: {target_stage2_samples:,.0f}")
        print(f"📊 Stage 2 expansion factor: {stage2_expansion_factor}x")
        
        # Create expanded Stage 2 with progress tracking
        print("🔄 Creating expanded Stage 2...")
        expanded_stage2 = []
        
        # Show progress for expansion
        for i in tqdm(range(stage2_expansion_factor), desc="📈 Expanding Stage 2", unit="cycles"):
            expanded_stage2.extend(original_stage2_lines)
            if (i + 1) % 5 == 0:  # Update every 5 cycles
                print(f"  📊 Progress: {len(expanded_stage2):,} samples, ~{len(expanded_stage2) * tokens_per_sample:,.0f} tokens")
        
        # Add random samples to reach exact target
        remaining_stage2 = int(target_stage2_samples) - len(expanded_stage2)
        if remaining_stage2 > 0:
            print(f"🔄 Adding {remaining_stage2:,} random samples to reach target...")
            expanded_stage2.extend(random.choices(original_stage2_lines, k=remaining_stage2))
        
        # Shuffle and save with progress
        print("🔄 Shuffling and saving Stage 2...")
        random.shuffle(expanded_stage2)
        
        print("💾 Writing Stage 2 file...")
        with open(stage2_file, 'w', encoding='utf-8') as f:
            # Write in chunks to show progress
            chunk_size = 100000  # 100K lines per chunk
            for i in tqdm(range(0, len(expanded_stage2), chunk_size), desc="💾 Writing Stage 2", unit="chunks"):
                chunk = expanded_stage2[i:i+chunk_size]
                f.write('\n'.join(chunk) + '\n')
        
        print(f"✅ Stage 2 expanded to {len(expanded_stage2):,} samples")
        print(f"📊 Estimated tokens: {len(expanded_stage2) * tokens_per_sample:,.0f}")
    
    print(f"\n🎉 Scaling complete!")
    print(f"📋 Next steps:")
    print(f"1. Retokenize Stage 1: python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage1 data/processed/cloud/stage1_mix.txt")
    print(f"2. Retokenize Stage 2: python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage2 data/processed/cloud/stage2_mix.txt")
    print(f"3. Start training: python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both --skip-data --resume")
    
    return True

def main():
    """Main function."""
    print("🚀 Scaling to actual 2B/5B token targets...")
    
    success = scale_up_to_target()
    
    if success:
        print("✅ Ready for retokenization!")
    else:
        print("❌ Scaling failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
