#!/usr/bin/env python3
"""Process downloaded data for cloud training."""

import argparse
import json
from pathlib import Path
from typing import Dict, List
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
    """Remove duplicate texts using MinHash."""
    seen_hashes = set()
    unique_texts = []
    
    for text in texts:
        # Simple hash for deduplication
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_texts.append(text)
    
    return unique_texts

def process_domain_data(domain_dir: Path, output_dir: Path, target_tokens: int):
    """Process domain-specific data."""
    print(f"Processing {domain_dir.name}...")
    
    all_texts = []
    for file_path in domain_dir.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.read().split('\n')
            all_texts.extend([clean_text(t) for t in texts if t.strip()])
    
    # Deduplicate
    unique_texts = deduplicate_texts(all_texts)
    
    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    current_tokens = sum(len(text) // 4 for text in unique_texts)
    
    if current_tokens >= target_tokens:
        # Sample to target
        import random
        random.shuffle(unique_texts)
        sampled_texts = []
        token_count = 0
        for text in unique_texts:
            if token_count >= target_tokens:
                break
            sampled_texts.append(text)
            token_count += len(text) // 4
        unique_texts = sampled_texts
    
    # Save processed data
    output_file = output_dir / f"{domain_dir.name}_processed.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(unique_texts))
    
    print(f"Saved {len(unique_texts)} texts ({current_tokens} tokens) to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process cloud training data")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/cloud"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/cloud"))
    parser.add_argument("--target-tokens", type=int, default=2_000_000_000)  # 2B tokens
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each domain
    for domain_dir in args.input_dir.iterdir():
        if domain_dir.is_dir():
            process_domain_data(domain_dir, args.output_dir, args.target_tokens)
    
    print("Data processing complete!")

if __name__ == "__main__":
    main()
