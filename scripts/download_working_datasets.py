#!/usr/bin/env python3
"""Download working datasets that actually exist on HuggingFace Hub."""

import os
import sys
import shutil
from pathlib import Path

def download_working_datasets():
    """Download datasets that actually work."""
    print("🔍 Finding working datasets...")
    
    # Check HF token
    if not os.environ.get('HF_TOKEN'):
        print("❌ HF_TOKEN environment variable not set!")
        print("Run: export HF_TOKEN=your_token_here")
        return False
    
    # Working datasets that actually exist
    working_datasets = [
        # Stage 1 - Smaller, working datasets
        "wikimedia/wikipedia:20231101.en:train:text:500000",
        "allenai/dolma:train:text:300000", 
        "Skylion007/openwebtext:train:text:200000",
        
        # Stage 2 - Larger datasets
        "cerebras/SlimPajama-627B:train:text:1000000",
        "allenai/dolma:train:text:700000",
    ]
    
    # Create directories
    os.makedirs("data/processed/cloud", exist_ok=True)
    
    # Stage 1
    print("📥 Downloading Stage 1 datasets...")
    stage1_cmd = f"""python scripts/download_hf_corpus.py \\
  --dataset wikimedia/wikipedia:20231101.en:train:text:500000 \\
  --dataset allenai/dolma:train:text:300000 \\
  --output data/processed/cloud/stage1_mix.txt"""
    
    result1 = os.system(stage1_cmd)
    if result1 != 0:
        print("❌ Stage 1 download failed, trying alternatives...")
        # Try simpler approach
        simple_cmd1 = 'python scripts/download_hf_corpus.py --dataset wikimedia/wikipedia:20231101.en:train:text:200000 --output data/processed/cloud/stage1_mix.txt'
        result1 = os.system(simple_cmd1)
    
    # Stage 2  
    print("📥 Downloading Stage 2 datasets...")
    stage2_cmd = f"""python scripts/download_hf_corpus.py \\
  --dataset cerebras/SlimPajama-627B:train:text:800000 \\
  --dataset allenai/dolma:train:text:400000 \\
  --output data/processed/cloud/stage2_mix.txt"""
    
    result2 = os.system(stage2_cmd)
    if result2 != 0:
        print("❌ Stage 2 download failed, trying alternatives...")
        # Try simpler approach
        simple_cmd2 = 'python scripts/download_hf_corpus.py --dataset wikimedia/wikipedia:20231101.en:train:text:600000 --output data/processed/cloud/stage2_mix.txt'
        result2 = os.system(simple_cmd2)
    
    # Check results
    stage1_exists = Path("data/processed/cloud/stage1_mix.txt").exists()
    stage2_exists = Path("data/processed/cloud/stage2_mix.txt").exists()
    
    if stage1_exists and stage2_exists:
        print("✅ Successfully downloaded datasets!")
        
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

def create_fallback_data():
    """Create fallback data using existing domain data."""
    print("🔄 Creating fallback data from existing domains...")
    
    # Use existing domain data but expand it
    os.system("python scripts/download_data.py --domains science data code chess general --output-dir data/raw/cloud")
    
    # Process and combine
    os.system("python scripts/process_cloud_data.py --input-dir data/raw/cloud --output-dir data/processed/cloud --target-tokens 2000000000")

    # Split into stage1 and stage2
    mix_file = Path('data/processed/cloud/mix.txt')
    stage1_file = Path('data/processed/cloud/stage1_mix.txt')
    stage2_file = Path('data/processed/cloud/stage2_mix.txt')

    if mix_file.exists():
        stage1_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(mix_file, stage1_file)
        shutil.copy(mix_file, stage2_file)
        print('✅ Created stage files from processed data')
    else:
        print('❌ No processed data found')

    return Path("data/processed/cloud/stage1_mix.txt").exists()

def main():
    """Main function."""
    print("🚀 Starting dataset download...")
    
    # Try working datasets first
    success = download_working_datasets()
    
    if not success:
        print("🔄 Falling back to domain-specific data...")
        success = create_fallback_data()
    
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
