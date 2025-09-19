#!/usr/bin/env python3
"""Download real 7B token datasets with proper error handling and compression support."""

import os
import sys
from pathlib import Path

def install_dependencies():
    """Install required dependencies for dataset downloading."""
    print("🔧 Installing required dependencies...")
    os.system("pip install zstandard py7zr")
    print("✅ Dependencies installed")

def download_stage1_data():
    """Download Stage 1 datasets (2B tokens)."""
    print("📥 Downloading Stage 1 datasets (2B tokens)...")
    
    # HF token should be set via environment variable
    if not os.environ.get('HF_TOKEN'):
        print("❌ HF_TOKEN environment variable not set!")
        print("Run: export HF_TOKEN=your_token_here")
        return False
    
    # Try working datasets in order of preference
    datasets_to_try = [
        # Working alternatives that don't require compression
        "openwebtext:train:text:1000000",
        "allenai/dolma-science:train:text:1000000", 
        "wikimedia/wikipedia:20231101.en:train:text:500000",
        
        # If RefinedWeb works with proper compression
        "huggingface/RefinedWeb:train:text:1000000",
    ]
    
    output_file = Path("data/processed/cloud/stage1_mix.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_samples = 0
    for dataset_spec in datasets_to_try:
        try:
            print(f"🔄 Trying dataset: {dataset_spec}")
            cmd = f'python scripts/download_hf_corpus.py --dataset {dataset_spec} --output {output_file}'
            result = os.system(cmd)
            if result == 0:
                print(f"✅ Successfully downloaded: {dataset_spec}")
                # Count samples
                with open(output_file, 'r') as f:
                    total_samples = sum(1 for _ in f)
                print(f"📊 Total samples in stage1: {total_samples:,}")
                break
            else:
                print(f"❌ Failed: {dataset_spec}")
        except Exception as e:
            print(f"❌ Error with {dataset_spec}: {e}")
    
    return total_samples > 0

def download_stage2_data():
    """Download Stage 2 datasets (5B tokens)."""
    print("📥 Downloading Stage 2 datasets (5B tokens)...")
    
    datasets_to_try = [
        # Working alternatives
        "cerebras/SlimPajama-627B:train:text:2000000",
        "allenai/dolma-books:train:text:1000000",
        "wikimedia/wikipedia:20231101.en:train:text:1000000",
    ]
    
    output_file = Path("data/processed/cloud/stage2_mix.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_samples = 0
    for dataset_spec in datasets_to_try:
        try:
            print(f"🔄 Trying dataset: {dataset_spec}")
            cmd = f'python scripts/download_hf_corpus.py --dataset {dataset_spec} --output {output_file}'
            result = os.system(cmd)
            if result == 0:
                print(f"✅ Successfully downloaded: {dataset_spec}")
                # Count samples
                with open(output_file, 'r') as f:
                    total_samples = sum(1 for _ in f)
                print(f"📊 Total samples in stage2: {total_samples:,}")
                break
            else:
                print(f"❌ Failed: {dataset_spec}")
        except Exception as e:
            print(f"❌ Error with {dataset_spec}: {e}")
    
    return total_samples > 0

def main():
    """Main download function."""
    print("🚀 Starting real dataset download for 7B token training...")
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    os.makedirs("data/processed/cloud", exist_ok=True)
    os.makedirs("data/tokenized/cloud", exist_ok=True)
    
    # Download datasets
    stage1_success = download_stage1_data()
    stage2_success = download_stage2_data()
    
    if stage1_success and stage2_success:
        print("🎉 Successfully downloaded all datasets!")
        print("📋 Next steps:")
        print("1. Process the data: python scripts/process_cloud_data.py --input-dir data/processed/cloud --output-dir data/processed/cloud")
        print("2. Retokenize: python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage1 data/processed/cloud/stage1_mix.txt")
        print("3. Start training: python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both --skip-data --resume")
    else:
        print("❌ Some datasets failed to download. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
