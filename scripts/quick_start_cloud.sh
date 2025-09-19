#!/bin/bash
# Quick start script for cloud training

set -e

echo "🚀 Starting FusterCluck Cloud Training..."

# Check if we're on a GPU instance
if ! nvidia-smi &> /dev/null; then
    echo "❌ No GPU detected. Please run on a GPU instance."
    exit 1
fi

# Set up environment
echo "📦 Setting up environment..."
bash scripts/setup_cloud_instance.sh

# Activate virtual environment
source .venv/bin/activate

# Download and process data
echo "📥 Downloading and processing data..."
python scripts/download_data.py --datasets refinedweb science --output-dir data/raw/cloud

# Process data
echo "🔄 Processing data..."
python scripts/process_cloud_data.py --input-dir data/raw/cloud --output-dir data/processed/cloud

# Build tokenizer
echo "🔤 Building tokenizer..."
python scripts/build_tokenizer.py data/processed/cloud --output artifacts/tokenizer/fustercluck --vocab-size 50000

# Tokenize data
echo "🎯 Tokenizing data..."
python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage1 data/processed/cloud/refinedweb_processed.txt
python scripts/pretokenize_text.py artifacts/tokenizer/fustercluck.model data/tokenized/cloud/stage2 data/processed/cloud/science_processed.txt

# Start training
echo "🏋️ Starting training..."
python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both

echo "✅ Training complete!"
