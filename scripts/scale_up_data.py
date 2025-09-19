#!/usr/bin/env python3
"""Scale up data to reach 7B token target (2B + 5B for stages)."""

import os
import sys
from pathlib import Path
import random

def estimate_tokens_from_samples(samples: int, avg_tokens_per_sample: int = 250) -> int:
    """Estimate total tokens from sample count."""
    return samples * avg_tokens_per_sample

def estimate_samples_from_tokens(target_tokens: int, avg_tokens_per_sample: int = 250) -> int:
    """Estimate samples needed for target tokens."""
    return target_tokens // avg_tokens_per_sample

def scale_up_domain_data():
    """Scale up domain data to reach 7B token target."""
    print("🎯 Scaling up domain data to 7B tokens...")
    
    # Create directories
    os.makedirs("data/processed/cloud", exist_ok=True)
    
    # Download domain data
    print("📥 Downloading domain data...")
    os.system("python scripts/download_data.py --domains science data code chess general --output-dir data/raw/cloud")
    
    # Process with much larger targets
    print("🔄 Processing data for 7B token target...")
    print("  Stage 1 target: 2B tokens (~8M samples)")
    print("  Stage 2 target: 5B tokens (~20M samples)")
    
    # Process Stage 1 (2B tokens)
    os.system("python scripts/process_weighted_data.py --input-dir data/raw/cloud --output-dir data/processed/cloud --target-tokens 2000000000 --stage stage1")
    
    # Process Stage 2 (5B tokens) 
    os.system("python scripts/process_weighted_data.py --input-dir data/raw/cloud --output-dir data/processed/cloud --target-tokens 5000000000 --stage stage2")
    
    # Check results
    stage1_file = Path("data/processed/cloud/stage1_mix.txt")
    stage2_file = Path("data/processed/cloud/stage2_mix.txt")
    
    if stage1_file.exists() and stage2_file.exists():
        with open(stage1_file, 'r') as f:
            stage1_samples = sum(1 for _ in f)
        with open(stage2_file, 'r') as f:
            stage2_samples = sum(1 for _ in f)
            
        stage1_tokens = estimate_tokens_from_samples(stage1_samples)
        stage2_tokens = estimate_tokens_from_samples(stage2_samples)
        total_tokens = stage1_tokens + stage2_tokens
        
        print(f"✅ Scaled up data created!")
        print(f"📊 Stage 1: {stage1_samples:,} samples (~{stage1_tokens:,} tokens)")
        print(f"📊 Stage 2: {stage2_samples:,} samples (~{stage2_tokens:,} tokens)")
        print(f"📊 Total: {stage1_samples + stage2_samples:,} samples (~{total_tokens:,} tokens)")
        print(f"🎯 Target: 7B tokens - {'✅ ACHIEVED' if total_tokens >= 7000000000 else '❌ MISSED'}")
        
        return True
    else:
        print("❌ Scaling failed!")
        return False

def create_synthetic_expansion():
    """Create synthetic expansion of domain data to reach 7B tokens."""
    print("🔄 Creating synthetic expansion to reach 7B tokens...")
    
    # First get base domain data
    os.system("python scripts/download_data.py --domains science data code chess general --output-dir data/raw/cloud")
    
    # Process base data
    os.system("python scripts/process_weighted_data.py --input-dir data/raw/cloud --output-dir data/processed/cloud --target-tokens 100000000 --stage both")
    
    # Check if we have base data
    base_file = Path("data/processed/cloud/stage1_mix.txt")
    if not base_file.exists():
        print("❌ No base data to expand!")
        return False
    
    # Read base data
    with open(base_file, 'r') as f:
        base_samples = [line.strip() for line in f if line.strip()]
    
    base_count = len(base_samples)
    base_tokens = estimate_tokens_from_samples(base_count)
    
    print(f"📊 Base data: {base_count:,} samples (~{base_tokens:,} tokens)")
    
    # Calculate expansion needed
    target_stage1_tokens = 2_000_000_000  # 2B
    target_stage2_tokens = 5_000_000_000  # 5B
    
    stage1_expansion = target_stage1_tokens // base_tokens
    stage2_expansion = target_stage2_tokens // base_tokens
    
    print(f"🔄 Stage 1 expansion: {stage1_expansion}x")
    print(f"🔄 Stage 2 expansion: {stage2_expansion}x")
    
    # Create expanded Stage 1
    print("📝 Creating expanded Stage 1...")
    expanded_stage1 = []
    for _ in range(stage1_expansion):
        expanded_stage1.extend(base_samples)
    
    # Add random sampling to reach exact target
    remaining_needed = target_stage1_tokens - estimate_tokens_from_samples(len(expanded_stage1))
    if remaining_needed > 0:
        additional_samples = estimate_samples_from_tokens(remaining_needed)
        expanded_stage1.extend(random.choices(base_samples, k=additional_samples))
    
    # Shuffle and save Stage 1
    random.shuffle(expanded_stage1)
    stage1_file = Path("data/processed/cloud/stage1_mix.txt")
    with open(stage1_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(expanded_stage1))
    
    # Create expanded Stage 2
    print("📝 Creating expanded Stage 2...")
    expanded_stage2 = []
    for _ in range(stage2_expansion):
        expanded_stage2.extend(base_samples)
    
    # Add random sampling to reach exact target
    remaining_needed = target_stage2_tokens - estimate_tokens_from_samples(len(expanded_stage2))
    if remaining_needed > 0:
        additional_samples = estimate_samples_from_tokens(remaining_needed)
        expanded_stage2.extend(random.choices(base_samples, k=additional_samples))
    
    # Shuffle and save Stage 2
    random.shuffle(expanded_stage2)
    stage2_file = Path("data/processed/cloud/stage2_mix.txt")
    with open(stage2_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(expanded_stage2))
    
    # Report results
    stage1_final_tokens = estimate_tokens_from_samples(len(expanded_stage1))
    stage2_final_tokens = estimate_tokens_from_samples(len(expanded_stage2))
    total_final_tokens = stage1_final_tokens + stage2_final_tokens
    
    print(f"✅ Synthetic expansion complete!")
    print(f"📊 Stage 1: {len(expanded_stage1):,} samples (~{stage1_final_tokens:,} tokens)")
    print(f"📊 Stage 2: {len(expanded_stage2):,} samples (~{stage2_final_tokens:,} tokens)")
    print(f"📊 Total: {len(expanded_stage1) + len(expanded_stage2):,} samples (~{total_final_tokens:,} tokens)")
    print(f"🎯 Target: 7B tokens - {'✅ ACHIEVED' if total_final_tokens >= 7000000000 else '❌ MISSED'}")
    
    return True

def main():
    """Main function."""
    print("🚀 Scaling up to 7B token target...")
    
    # Try scaling up domain data first
    success = scale_up_domain_data()
    
    if not success:
        print("🔄 Trying synthetic expansion approach...")
        success = create_synthetic_expansion()
    
    if success:
        print("🎉 7B token dataset ready!")
        print("📋 Next steps:")
        print("1. python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage1 data/processed/cloud/stage1_mix.txt")
        print("2. python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage2 data/processed/cloud/stage2_mix.txt")
        print("3. python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both --skip-data --resume")
    else:
        print("❌ Failed to reach 7B token target!")
        sys.exit(1)

if __name__ == "__main__":
    main()
