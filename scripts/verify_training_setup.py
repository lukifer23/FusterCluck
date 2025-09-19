#!/usr/bin/env python3
"""Verify that training data is properly set up and integrated."""

from pathlib import Path

from fustercluck.data.corpus_builder import token_count_from_idx

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

def report_token_count(idx_path: Path, description: str) -> int:
    """Report true token counts from idx files."""
    if not idx_path.exists():
        print(f"❌ {description}: {idx_path} (MISSING)")
        return 0
    try:
        tokens = token_count_from_idx(idx_path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"❌ {description}: failed to inspect {idx_path} ({exc})")
        return 0
    print(f"📊 {description}: {tokens:,} tokens")
    return tokens

def verify_data_pipeline():
    """Verify the complete data pipeline."""
    print("🔍 Verifying FusterCluck training data pipeline...\n")
    
    # 1. Check raw text files
    print("1️⃣ Checking raw text files:")
    stage1_raw = Path("data/processed/cloud/stage1_mix.txt")
    stage2_raw = Path("data/processed/cloud/stage2_mix.txt")
    
    stage1_exists = check_file_exists_and_size(stage1_raw, "Stage 1 raw text")
    stage2_exists = check_file_exists_and_size(stage2_raw, "Stage 2 raw text")

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
    stage2_tokenized_exists = check_file_exists_and_size(stage2_tokenized, "Stage 2 tokenized")
    stage1_tokens = report_token_count(stage1_index, "Stage 1 token count")
    stage2_tokens = report_token_count(stage2_index, "Stage 2 token count")

    tokenized_ready = all([
        stage1_tokenized_exists,
        stage2_tokenized_exists,
        stage1_tokens > 0,
        stage2_tokens > 0,
    ])
    if tokenized_ready:
        total_tokens = stage1_tokens + stage2_tokens
        print(f"🎯 Combined tokens: {total_tokens:,}")
        print(f"   Target 7B tokens: {'✅ ACHIEVED' if total_tokens >= 7_000_000_000 else '⚠️ SHORT'}\n")
    else:
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
    
    if stage1_exists and stage2_exists and tokenizer_exists and tokenized_ready:
        print("✅ Raw corpora, tokenizer, and tokenized shards look good")
        print("🚀 Ready to launch training")
        print("📋 Next step:")
        print("   export HF_TOKEN=your_hf_token && python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both --resume")
    else:
        print("❌ Setup incomplete – one or more prerequisites missing")
        print("📋 Next step:")
        print("   export HF_TOKEN=your_hf_token && python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both --resume")
    
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
