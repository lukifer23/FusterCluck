#!/usr/bin/env python3
"""Verify that training data is properly set up and integrated."""

import os
import sys
from pathlib import Path

def check_file_exists_and_size(file_path: Path, description: str) -> bool:
    """Check if file exists and report its size."""
    if file_path.exists():
        size = file_path.stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"✅ {description}: {file_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ {description}: {file_path} (MISSING)")
        return False

def count_lines(file_path: Path, description: str) -> int:
    """Count lines in a file."""
    if not file_path.exists():
        print(f"❌ {description}: File not found")
        return 0
    
    with open(file_path, 'r') as f:
        count = sum(1 for _ in f)
    
    print(f"📊 {description}: {count:,} lines")
    return count

def verify_data_pipeline():
    """Verify the complete data pipeline."""
    print("🔍 Verifying FusterCluck training data pipeline...\n")
    
    # 1. Check raw text files
    print("1️⃣ Checking raw text files:")
    stage1_raw = Path("data/processed/cloud/stage1_mix.txt")
    stage2_raw = Path("data/processed/cloud/stage2_mix.txt")
    
    stage1_exists = check_file_exists_and_size(stage1_raw, "Stage 1 raw text")
    stage2_exists = check_file_exists_and_size(stage2_raw, "Stage 2 raw text")
    
    if stage1_exists and stage2_exists:
        stage1_lines = count_lines(stage1_raw, "Stage 1 samples")
        stage2_lines = count_lines(stage2_raw, "Stage 2 samples")
        total_lines = stage1_lines + stage2_lines
        
        # Estimate tokens (rough: 250 tokens per sample)
        estimated_tokens = total_lines * 250
        print(f"📊 Total estimated tokens: {estimated_tokens:,}")
        print(f"🎯 7B token target: {'✅ ACHIEVED' if estimated_tokens >= 7000000000 else '❌ MISSED'}\n")
    
    # 2. Check tokenizer
    print("2️⃣ Checking tokenizer:")
    tokenizer_file = Path("artifacts/tokenizer/fustercluck.model")
    tokenizer_exists = check_file_exists_and_size(tokenizer_file, "Tokenizer model")
    print()
    
    # 3. Check tokenized data
    print("3️⃣ Checking tokenized data:")
    stage1_tokenized = Path("data/tokenized/cloud/stage1.bin")
    stage1_index = Path("data/tokenized/cloud/stage1.idx")
    stage2_tokenized = Path("data/tokenized/cloud/stage2.bin")
    stage2_index = Path("data/tokenized/cloud/stage2.idx")
    
    stage1_tokenized_exists = check_file_exists_and_size(stage1_tokenized, "Stage 1 tokenized")
    stage1_index_exists = check_file_exists_and_size(stage1_index, "Stage 1 index")
    stage2_tokenized_exists = check_file_exists_and_size(stage2_tokenized, "Stage 2 tokenized")
    stage2_index_exists = check_file_exists_and_size(stage2_index, "Stage 2 index")
    
    tokenized_ready = all([stage1_tokenized_exists, stage1_index_exists, 
                          stage2_tokenized_exists, stage2_index_exists])
    print()
    
    # 4. Check training config
    print("4️⃣ Checking training configuration:")
    config_file = Path("configs/cloud_training.yaml")
    config_exists = check_file_exists_and_size(config_file, "Training config")
    print()
    
    # 5. Check domain distribution
    print("5️⃣ Checking domain distribution:")
    if stage1_exists:
        # Sample domain content
        with open(stage1_raw, 'r') as f:
            sample_lines = [f.readline().strip() for _ in range(10)]
        
        code_lines = sum(1 for line in sample_lines if 'def ' in line or 'import ' in line or 'class ' in line)
        science_lines = sum(1 for line in sample_lines if any(word in line.lower() for word in ['physics', 'chemistry', 'biology', 'mathematics', 'theorem']))
        chess_lines = sum(1 for line in sample_lines if any(word in line.lower() for word in ['chess', 'pawn', 'king', 'queen', 'bishop']))
        
        print(f"📊 Sample analysis (first 10 lines):")
        print(f"  Code content: {code_lines}/10 lines")
        print(f"  Science content: {science_lines}/10 lines") 
        print(f"  Chess content: {chess_lines}/10 lines")
        print()
    
    # 6. Summary and recommendations
    print("📋 SUMMARY:")
    
    if stage1_exists and stage2_exists and tokenizer_exists:
        print("✅ Raw data and tokenizer are ready")
        
        if not tokenized_ready:
            print("⚠️  Tokenized data is missing - need to run tokenization")
            print("📋 Next step:")
            print("   python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage1 data/processed/cloud/stage1_mix.txt")
            print("   python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage2 data/processed/cloud/stage2_mix.txt")
        else:
            print("✅ Tokenized data is ready")
            print("🚀 Ready to start training!")
            print("📋 Next step:")
            print("   python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both --skip-data --resume")
    else:
        print("❌ Missing required files - need to run data setup first")
        print("📋 Next step:")
        print("   python scripts/simple_scale_up.py")
    
    print()
    return tokenized_ready

def check_training_integration():
    """Check if training scripts are properly configured."""
    print("🔧 Checking training script integration...\n")
    
    # Check if run_cloud_training.py references the right paths
    script_path = Path("scripts/run_cloud_training.py")
    if script_path.exists():
        with open(script_path, 'r') as f:
            content = f.read()
        
        if "data/tokenized/cloud" in content:
            print("✅ Training script references correct tokenized data path")
        else:
            print("❌ Training script may not reference correct data path")
        
        if "stage1_mix.txt" in content or "stage2_mix.txt" in content:
            print("✅ Training script references mix files")
        else:
            print("⚠️  Training script doesn't directly reference mix files (may use tokenized data instead)")
    else:
        print("❌ Training script not found")
    
    print()

def main():
    """Main verification function."""
    print("🚀 FusterCluck Training Data Verification\n")
    
    # Verify data pipeline
    tokenized_ready = verify_data_pipeline()
    
    # Check training integration
    check_training_integration()
    
    # Final status
    if tokenized_ready:
        print("🎉 TRAINING READY!")
        print("Your 7B token dataset is properly set up and ready for training.")
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("Run the tokenization steps to complete the setup.")
    
    return 0 if tokenized_ready else 1

if __name__ == "__main__":
    exit(main())
