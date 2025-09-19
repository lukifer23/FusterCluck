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

# Run managed training pipeline (handles data download, processing, tokenization)
echo "🏋️ Starting training pipeline..."
python scripts/run_cloud_training.py --config configs/cloud_training.yaml --stage both

echo "✅ Training complete!"
