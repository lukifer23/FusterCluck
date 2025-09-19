#!/usr/bin/env python3
"""Manual data setup using existing domain data - guaranteed to work."""

import os
import shutil
from pathlib import Path

def setup_manual_data():
    """Set up data using existing domain downloads."""
    print("🔧 Setting up manual data from existing domains...")
    
    # Create directories
    os.makedirs("data/processed/cloud", exist_ok=True)
    os.makedirs("data/tokenized/cloud", exist_ok=True)
    
    # Download existing domain data
    print("📥 Downloading domain data...")
    os.system("python scripts/download_data.py --domains science data code chess general --output-dir data/raw/cloud")
    
    # Process the data
    print("🔄 Processing data...")
    os.system("python scripts/process_cloud_data.py --input-dir data/raw/cloud --output-dir data/processed/cloud --target-tokens 100000000")
    
    # Check if processing created mix.txt
    mix_file = Path("data/processed/cloud/mix.txt")
    if mix_file.exists():
        print("✅ Processing successful!")
        
        # Copy to stage files
        stage1_file = Path("data/processed/cloud/stage1_mix.txt")
        stage2_file = Path("data/processed/cloud/stage2_mix.txt")
        
        shutil.copy(mix_file, stage1_file)
        shutil.copy(mix_file, stage2_file)
        
        # Count samples
        with open(stage1_file, 'r') as f:
            count = sum(1 for _ in f)
        
        print(f"📊 Created stage files with {count:,} samples each")
        print(f"📁 Stage 1: {stage1_file}")
        print(f"📁 Stage 2: {stage2_file}")
        
        return True
    else:
        print("❌ Processing failed - no mix.txt created")
        return False

def main():
    """Main function."""
    success = setup_manual_data()
    
    if success:
        print("🎉 Manual data setup complete!")
        print("📋 Next steps:")
        print("1. python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage1 data/processed/cloud/stage1_mix.txt")
        print("2. python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage2 data/processed/cloud/stage2_mix.txt")
        print("3. python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both --skip-data --resume")
    else:
        print("❌ Manual setup failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
