#!/usr/bin/env python3
"""Smart data download that respects storage limits and focuses on quality over quantity."""

import os
import sys
from pathlib import Path

def estimate_storage_needed(samples: int) -> float:
    """Estimate storage needed in GB."""
    # Rough estimate: 1KB per sample average
    return (samples * 1024) / (1024**3)  # Convert to GB

def download_reasonable_datasets():
    """Download datasets that fit within 2.2TB storage limit."""
    print("🎯 Downloading reasonable datasets for 2.2TB storage...")
    
    # Set HF token
    if not os.environ.get('HF_TOKEN'):
        print("❌ HF_TOKEN environment variable not set!")
        return False
    
    # Much smaller, reasonable datasets
    stage1_datasets = [
        # Wikipedia is reliable and reasonable size
        "wikimedia/wikipedia:20231101.en:train:text:100000",  # ~100K samples
    ]
    
    stage2_datasets = [
        # Use existing domain data + small HF datasets
        "wikimedia/wikipedia:20231101.en:train:text:200000",  # ~200K samples
    ]
    
    print("📊 Estimated storage needs:")
    print(f"  Stage 1: {estimate_storage_needed(100000):.1f} GB")
    print(f"  Stage 2: {estimate_storage_needed(200000):.1f} GB")
    print(f"  Total: {estimate_storage_needed(300000):.1f} GB (well under 2.2TB limit)")
    
    # Create directories
    os.makedirs("data/processed/cloud", exist_ok=True)
    
    # Download Stage 1
    print("\n📥 Downloading Stage 1...")
    stage1_cmd = 'python scripts/download_hf_corpus.py --dataset wikimedia/wikipedia:20231101.en:train:text:100000 --output data/processed/cloud/stage1_mix.txt'
    result1 = os.system(stage1_cmd)
    
    # Download Stage 2
    print("\n📥 Downloading Stage 2...")
    stage2_cmd = 'python scripts/download_hf_corpus.py --dataset wikimedia/wikipedia:20231101.en:train:text:200000 --output data/processed/cloud/stage2_mix.txt'
    result2 = os.system(stage2_cmd)
    
    # Check results
    stage1_exists = Path("data/processed/cloud/stage1_mix.txt").exists()
    stage2_exists = Path("data/processed/cloud/stage2_mix.txt").exists()
    
    if stage1_exists and stage2_exists:
        print("✅ Successfully downloaded reasonable datasets!")
        
        # Count samples
        with open("data/processed/cloud/stage1_mix.txt", 'r') as f:
            stage1_count = sum(1 for _ in f)
        with open("data/processed/cloud/stage2_mix.txt", 'r') as f:
            stage2_count = sum(1 for _ in f)
            
        print(f"📊 Stage 1 samples: {stage1_count:,}")
        print(f"📊 Stage 2 samples: {stage2_count:,}")
        print(f"📊 Total samples: {stage1_count + stage2_count:,}")
        
        return True
    else:
        print("❌ Dataset download failed!")
        return False

def create_domain_focused_data():
    """Create data focused on our domains with proper weighting."""
    print("🎯 Creating domain-focused data...")
    
    # Download domain data (this is small and reliable)
    os.system("python scripts/download_data.py --domains science data code chess general --output-dir data/raw/cloud")
    
    # Process with proper weighting
    os.system("python scripts/process_weighted_data.py --input-dir data/raw/cloud --output-dir data/processed/cloud --target-tokens 50000000 --stage both")
    
    # Check results
    stage1_file = Path("data/processed/cloud/stage1_mix.txt")
    stage2_file = Path("data/processed/cloud/stage2_mix.txt")
    
    if stage1_file.exists() and stage2_file.exists():
        with open(stage1_file, 'r') as f:
            stage1_count = sum(1 for _ in f)
        with open(stage2_file, 'r') as f:
            stage2_count = sum(1 for _ in f)
            
        print(f"✅ Domain-focused data created!")
        print(f"📊 Stage 1: {stage1_count:,} samples")
        print(f"📊 Stage 2: {stage2_count:,} samples")
        print(f"🎯 Focus: Science (35%), Code (25%), Data (20%), Chess (15%), General (5%)")
        
        return True
    else:
        print("❌ Domain data creation failed!")
        return False

def main():
    """Main function."""
    print("🚀 Smart data download for 2.2TB storage limit...")
    
    # Try reasonable HF datasets first
    success = download_reasonable_datasets()
    
    if not success:
        print("🔄 Falling back to domain-focused data...")
        success = create_domain_focused_data()
    
    if success:
        print("🎉 Data ready for training!")
        print("📋 Next steps:")
        print("1. python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage1 data/processed/cloud/stage1_mix.txt")
        print("2. python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage2 data/processed/cloud/stage2_mix.txt")
        print("3. python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both --skip-data --resume")
    else:
        print("❌ All download methods failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
