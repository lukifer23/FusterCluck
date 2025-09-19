#!/usr/bin/env python3
"""Download and prepare training datasets for cloud training."""

import os
import subprocess
import argparse
from pathlib import Path
from datasets import load_dataset
import requests
import tarfile
import zipfile

def download_huggingface_dataset(dataset_name: str, output_dir: Path, subset: str = None):
    """Download dataset from HuggingFace."""
    print(f"Downloading {dataset_name}...")
    dataset = load_dataset(dataset_name, subset, streaming=True)
    
    output_path = output_dir / dataset_name.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as text files
    for split_name, split_data in dataset.items():
        split_path = output_path / f"{split_name}.txt"
        with open(split_path, 'w', encoding='utf-8') as f:
            for i, example in enumerate(split_data):
                if i >= 1000000:  # Limit for demo
                    break
                f.write(example['text'] + '\n')
        print(f"Saved {split_name} to {split_path}")

def download_refinedweb(output_dir: Path):
    """Download RefinedWeb dataset."""
    print("Downloading RefinedWeb...")
    # Use streaming to avoid memory issues
    dataset = load_dataset("huggingface/refinedweb", "default", streaming=True)
    
    output_path = output_dir / "refinedweb"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download in chunks
    chunk_size = 10000
    for i, example in enumerate(dataset['train']):
        if i >= 1000000:  # Limit for demo
            break
        chunk_file = output_path / f"chunk_{i // chunk_size:04d}.txt"
        with open(chunk_file, 'a', encoding='utf-8') as f:
            f.write(example['text'] + '\n')

def download_science_data(output_dir: Path):
    """Download science-specific datasets."""
    science_dir = output_dir / "science"
    science_dir.mkdir(parents=True, exist_ok=True)
    
    # ArXiv abstracts
    print("Downloading ArXiv abstracts...")
    dataset = load_dataset("scientific_papers", "arxiv", streaming=True)
    
    with open(science_dir / "arxiv.txt", 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset['train']):
            if i >= 100000:
                break
            f.write(example['article'] + '\n')

def main():
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/cloud"))
    parser.add_argument("--datasets", nargs="+", 
                       choices=["refinedweb", "slimpajama", "dolma", "science", "code", "all"],
                       default=["refinedweb", "science"])
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if "refinedweb" in args.datasets or "all" in args.datasets:
        download_refinedweb(args.output_dir)
    
    if "science" in args.datasets or "all" in args.datasets:
        download_science_data(args.output_dir)
    
    print("Data download complete!")

if __name__ == "__main__":
    main()
