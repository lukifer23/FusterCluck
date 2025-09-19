#!/usr/bin/env python3
"""Fast tokenization script that processes data in chunks to avoid hanging."""

import argparse
import logging
from pathlib import Path
import numpy as np
import sentencepiece as spm
from tqdm import tqdm
import sys

def tokenize_file_chunked(
    tokenizer_path: Path,
    input_file: Path, 
    output_prefix: Path,
    chunk_size: int = 100000,  # Process 100K lines at a time
    minimum_chars: int = 1
) -> tuple[int, int]:
    """Tokenize a file in chunks to avoid memory issues."""
    
    print(f"🔤 Loading tokenizer from {tokenizer_path}")
    processor = spm.SentencePieceProcessor()
    processor.load(str(tokenizer_path))
    
    eos_id = processor.eos_id()
    if eos_id < 0:
        raise RuntimeError("SentencePiece model is missing an EOS token")
    
    bin_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Tokenizing {input_file} -> {output_prefix}")
    print(f"📊 Processing in chunks of {chunk_size:,} lines")
    
    total_tokens = 0
    total_sequences = 0
    offset_bytes = 0
    
    # Count total lines first
    print("🔍 Counting lines...")
    with input_file.open("r", encoding="utf-8", errors="ignore") as f:
        total_lines = sum(1 for _ in f)
    print(f"📊 Total lines: {total_lines:,}")
    
    with bin_path.open("wb") as bin_handle, idx_path.open("wb") as idx_handle:
        # Write initial offset
        idx_handle.write(np.asarray([0], dtype=np.int64).tobytes())
        
        with input_file.open("r", encoding="utf-8", errors="ignore") as input_handle:
            chunk_count = 0
            current_chunk = []
            
            for line_num, line in enumerate(tqdm(input_handle, total=total_lines, desc="Tokenizing")):
                line = line.strip()
                if len(line) < minimum_chars:
                    continue
                
                current_chunk.append(line)
                
                # Process chunk when it reaches chunk_size
                if len(current_chunk) >= chunk_size or line_num == total_lines - 1:
                    chunk_count += 1
                    print(f"🔄 Processing chunk {chunk_count} ({len(current_chunk)} lines)")
                    
                    for line in current_chunk:
                        pieces = processor.EncodeAsIds(line)
                        if not pieces:
                            continue
                        
                        # Add EOS token
                        pieces.append(eos_id)
                        
                        # Write tokens as uint16
                        token_array = np.asarray(pieces, dtype=np.uint16)
                        bin_handle.write(token_array.tobytes())
                        
                        # Update index
                        offset_bytes += len(token_array) * 2  # 2 bytes per uint16
                        idx_handle.write(np.asarray([offset_bytes], dtype=np.int64).tobytes())
                        
                        total_tokens += len(token_array)
                        total_sequences += 1
                    
                    current_chunk = []
                    
                    # Progress update
                    if chunk_count % 10 == 0:
                        print(f"📊 Progress: {line_num+1:,}/{total_lines:,} lines, {total_sequences:,} sequences, {total_tokens:,} tokens")
    
    print(f"✅ Tokenization complete!")
    print(f"📊 Total sequences: {total_sequences:,}")
    print(f"📊 Total tokens: {total_tokens:,}")
    print(f"📁 Output files:")
    print(f"  - {bin_path} ({bin_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"  - {idx_path}")
    
    return total_tokens, total_sequences

def main():
    parser = argparse.ArgumentParser(description="Fast tokenization with chunking")
    parser.add_argument("tokenizer_path", type=Path, help="Path to tokenizer model")
    parser.add_argument("output_prefix", type=Path, help="Output prefix for .bin/.idx files")
    parser.add_argument("input_file", type=Path, help="Input text file")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Lines per chunk")
    parser.add_argument("--minimum-chars", type=int, default=1, help="Minimum characters per line")
    
    args = parser.parse_args()
    
    if not args.tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {args.tokenizer_path}")
        sys.exit(1)
    
    if not args.input_file.exists():
        print(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)
    
    try:
        total_tokens, total_sequences = tokenize_file_chunked(
            args.tokenizer_path,
            args.input_file,
            args.output_prefix,
            chunk_size=args.chunk_size,
            minimum_chars=args.minimum_chars
        )
        
        print(f"🎉 Success! Tokenized {total_sequences:,} sequences into {total_tokens:,} tokens")
        
    except KeyboardInterrupt:
        print("\n⚠️  Tokenization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
