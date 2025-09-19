#!/usr/bin/env python3
"""Process data with proper domain weighting for code/science/math/chess focus."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import re

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Normalize unicode
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    return text.strip()

def deduplicate_texts(texts: List[str]) -> List[str]:
    """Remove duplicate texts using hash."""
    seen_hashes = set()
    unique_texts = []
    
    for text in texts:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_texts.append(text)
    
    return unique_texts

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
    return len(text) // 4

TEXT_EXTENSIONS = {".txt", ".rst", ".md", ".mdx", ".py", ".pgn", ".jsonl"}

def load_domain_weights(config_path: Path) -> Dict[str, float]:
    """Load domain weights from config."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    weights = {}
    for source in config['sources']:
        weights[source['name']] = source['weight']
    
    return weights

def collect_domain_data(domain_dir: Path) -> List[str]:
    """Collect all text from a domain directory."""
    all_texts = []
    
    for path in domain_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
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
    
    return all_texts

def create_weighted_mix(domain_data: Dict[str, List[str]], weights: Dict[str, float], target_tokens: int) -> List[str]:
    """Create weighted mix of domain data."""
    print("🎯 Creating weighted domain mix...")
    
    # Calculate target tokens per domain
    domain_targets = {}
    for domain, weight in weights.items():
        domain_targets[domain] = int(target_tokens * weight)
        print(f"  {domain}: {weight:.1%} = {domain_targets[domain]:,} tokens")
    
    mixed_texts = []
    
    for domain, texts in domain_data.items():
        if domain not in weights:
            continue
            
        target = domain_targets[domain]
        if not texts:
            print(f"⚠️  No data found for {domain}")
            continue
        
        # Clean and deduplicate domain texts
        cleaned_texts = [clean_text(text) for text in texts if clean_text(text)]
        unique_texts = deduplicate_texts(cleaned_texts)
        
        # Estimate current tokens
        current_tokens = sum(estimate_tokens(text) for text in unique_texts)
        
        if current_tokens >= target:
            # Sample to target
            random.shuffle(unique_texts)
            sampled_texts = []
            token_count = 0
            for text in unique_texts:
                if token_count >= target:
                    break
                sampled_texts.append(text)
                token_count += estimate_tokens(text)
            
            mixed_texts.extend(sampled_texts)
            print(f"  ✅ {domain}: sampled {len(sampled_texts)} texts ({token_count:,} tokens)")
        else:
            # Use all available
            mixed_texts.extend(unique_texts)
            print(f"  ✅ {domain}: used all {len(unique_texts)} texts ({current_tokens:,} tokens)")
    
    # Shuffle the final mix
    random.shuffle(mixed_texts)
    
    total_tokens = sum(estimate_tokens(text) for text in mixed_texts)
    print(f"🎉 Final mix: {len(mixed_texts)} texts, {total_tokens:,} tokens")
    
    return mixed_texts

def main():
    parser = argparse.ArgumentParser(description="Process data with domain weighting")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/cloud"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/cloud"))
    parser.add_argument("--config", type=Path, default=Path("configs/stage0_sources.json"))
    parser.add_argument("--target-tokens", type=int, default=2_000_000_000)  # 2B tokens
    parser.add_argument("--stage", choices=["stage1", "stage2", "both"], default="both")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load domain weights
    weights = load_domain_weights(args.config)
    print(f"📊 Domain weights: {weights}")
    
    # Collect data from all domains
    domain_data = {}
    for domain_dir in args.input_dir.iterdir():
        if domain_dir.is_dir():
            print(f"📁 Collecting data from {domain_dir.name}...")
            domain_data[domain_dir.name] = collect_domain_data(domain_dir)
            print(f"  Found {len(domain_data[domain_dir.name])} texts")
    
    # Create weighted mixes for both stages
    if args.stage in ["stage1", "both"]:
        print("\n🚀 Creating Stage 1 mix (2B tokens)...")
        stage1_tokens = args.target_tokens
        stage1_mix = create_weighted_mix(domain_data, weights, stage1_tokens)
        
        stage1_file = args.output_dir / "stage1_mix.txt"
        with open(stage1_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(stage1_mix))
        print(f"💾 Saved Stage 1 mix to {stage1_file}")
    
    if args.stage in ["stage2", "both"]:
        print("\n🚀 Creating Stage 2 mix (5B tokens)...")
        stage2_tokens = int(args.target_tokens * 2.5)  # 5B tokens
        stage2_mix = create_weighted_mix(domain_data, weights, stage2_tokens)
        
        stage2_file = args.output_dir / "stage2_mix.txt"
        with open(stage2_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(stage2_mix))
        print(f"💾 Saved Stage 2 mix to {stage2_file}")
    
    print("\n✅ Weighted data processing complete!")
    print(f"📋 Domain focus: Science ({weights['science']:.1%}), Code ({weights['code']:.1%}), Data ({weights['data']:.1%}), Chess ({weights['chess']:.1%}), General ({weights['general']:.1%})")

if __name__ == "__main__":
    main()
