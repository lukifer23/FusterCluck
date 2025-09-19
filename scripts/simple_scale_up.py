#!/usr/bin/env python3
"""Simple scaling script that doesn't rely on config files."""

import os
import sys
from pathlib import Path
import random
import json

def estimate_tokens_from_samples(samples: int, avg_tokens_per_sample: int = 250) -> int:
    """Estimate total tokens from sample count."""
    return samples * avg_tokens_per_sample

def estimate_samples_from_tokens(target_tokens: int, avg_tokens_per_sample: int = 250) -> int:
    """Estimate samples needed for target tokens."""
    return target_tokens // avg_tokens_per_sample

def collect_all_domain_data(input_dir: Path) -> dict:
    """Collect all text from domain directories."""
    domain_data = {}
    text_extensions = {".txt", ".rst", ".md", ".mdx", ".py", ".pgn", ".jsonl"}
    
    for domain_dir in input_dir.iterdir():
        if domain_dir.is_dir():
            print(f"📁 Collecting data from {domain_dir.name}...")
            all_texts = []
            
            for path in domain_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in text_extensions:
                    if path.suffix.lower() == ".jsonl":
                        # Handle JSONL files
                        with path.open("r", encoding="utf-8") as handle:
                            for line in handle:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    record = json.loads(line)
                                    instruction = record.get("instruction") or record.get("prompt") or ""
                                    response = record.get("response") or record.get("output") or ""
                                    role = record.get("context") or ""
                                    combined = "\n\n".join(part for part in [instruction, role, response] if part)
                                    if combined:
                                        all_texts.append(combined)
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # Handle regular text files
                        with path.open("r", encoding="utf-8", errors="ignore") as handle:
                            for line in handle:
                                line = line.strip()
                                if line:
                                    all_texts.append(line)
            
            domain_data[domain_dir.name] = all_texts
            print(f"  Found {len(all_texts)} texts")
    
    return domain_data

def create_weighted_expansion(domain_data: dict, target_tokens: int) -> list:
    """Create weighted expansion to reach target tokens."""
    # Define weights directly (no config file needed)
    weights = {
        "science": 0.35,
        "code": 0.25, 
        "data": 0.20,
        "chess": 0.15,
        "general": 0.05
    }
    
    print("🎯 Creating weighted expansion...")
    print(f"  Science: {weights['science']:.1%}")
    print(f"  Code: {weights['code']:.1%}")
    print(f"  Data: {weights['data']:.1%}")
    print(f"  Chess: {weights['chess']:.1%}")
    print(f"  General: {weights['general']:.1%}")
    
    # Calculate target tokens per domain
    domain_targets = {}
    for domain, weight in weights.items():
        domain_targets[domain] = int(target_tokens * weight)
        print(f"  {domain}: {domain_targets[domain]:,} tokens")
    
    expanded_texts = []
    
    for domain, texts in domain_data.items():
        if domain not in weights or not texts:
            continue
            
        target = domain_targets[domain]
        current_tokens = estimate_tokens_from_samples(len(texts))
        
        if current_tokens >= target:
            # Sample to target
            random.shuffle(texts)
            sampled_texts = []
            token_count = 0
            for text in texts:
                if token_count >= target:
                    break
                sampled_texts.append(text)
                token_count += estimate_tokens_from_samples(1)
            
            expanded_texts.extend(sampled_texts)
            print(f"  ✅ {domain}: sampled {len(sampled_texts)} texts")
        else:
            # Expand to reach target
            expansion_factor = target // current_tokens
            if expansion_factor < 1:
                expansion_factor = 1
            
            expanded_domain = []
            for _ in range(expansion_factor):
                expanded_domain.extend(texts)
            
            # Add random samples to reach exact target
            remaining_needed = target - estimate_tokens_from_samples(len(expanded_domain))
            if remaining_needed > 0:
                additional_samples = estimate_samples_from_tokens(remaining_needed)
                expanded_domain.extend(random.choices(texts, k=additional_samples))
            
            expanded_texts.extend(expanded_domain)
            print(f"  ✅ {domain}: expanded to {len(expanded_domain)} texts")
    
    # Shuffle the final mix
    random.shuffle(expanded_texts)
    
    total_tokens = estimate_tokens_from_samples(len(expanded_texts))
    print(f"🎉 Final mix: {len(expanded_texts)} texts, ~{total_tokens:,} tokens")
    
    return expanded_texts

def main():
    """Main function."""
    print("🚀 Simple scaling to 7B tokens...")
    
    # Create directories
    input_dir = Path("data/raw/cloud")
    output_dir = Path("data/processed/cloud")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download domain data if needed
    if not input_dir.exists() or not any(input_dir.iterdir()):
        print("📥 Downloading domain data...")
        os.system("python scripts/download_data.py --domains science data code chess general --output-dir data/raw/cloud")
    
    # Collect domain data
    domain_data = collect_all_domain_data(input_dir)
    
    if not domain_data:
        print("❌ No domain data found!")
        return 1
    
    # Create Stage 1 (2B tokens)
    print("\n🚀 Creating Stage 1 (2B tokens)...")
    stage1_texts = create_weighted_expansion(domain_data, 2_000_000_000)
    
    stage1_file = output_dir / "stage1_mix.txt"
    with open(stage1_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stage1_texts))
    
    # Create Stage 2 (5B tokens)
    print("\n🚀 Creating Stage 2 (5B tokens)...")
    stage2_texts = create_weighted_expansion(domain_data, 5_000_000_000)
    
    stage2_file = output_dir / "stage2_mix.txt"
    with open(stage2_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stage2_texts))
    
    # Report results
    stage1_tokens = estimate_tokens_from_samples(len(stage1_texts))
    stage2_tokens = estimate_tokens_from_samples(len(stage2_texts))
    total_tokens = stage1_tokens + stage2_tokens
    
    print(f"\n✅ Scaling complete!")
    print(f"📊 Stage 1: {len(stage1_texts):,} samples (~{stage1_tokens:,} tokens)")
    print(f"📊 Stage 2: {len(stage2_texts):,} samples (~{stage2_tokens:,} tokens)")
    print(f"📊 Total: {len(stage1_texts) + len(stage2_texts):,} samples (~{total_tokens:,} tokens)")
    print(f"🎯 Target: 7B tokens - {'✅ ACHIEVED' if total_tokens >= 7000000000 else '❌ MISSED'}")
    
    print(f"\n📋 Next steps:")
    print(f"1. python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage1 data/processed/cloud/stage1_mix.txt")
    print(f"2. python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage2 data/processed/cloud/stage2_mix.txt")
    print(f"3. python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both --skip-data --resume")
    
    return 0

if __name__ == "__main__":
    exit(main())
