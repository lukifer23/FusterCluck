#!/bin/bash
# One-command RunPod setup and training start

set -e

echo "🚀 FusterCluck RunPod Quick Start"
echo "================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Please run this from the FusterCluck root directory"
    exit 1
fi

# Check GPU
echo "🎮 Checking GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "❌ No GPU detected. Please run on a GPU instance."
    exit 1
fi

nvidia-smi

# Check disk space
echo "💾 Checking disk space..."
df -h /workspace

# Run setup
echo "📦 Running RunPod setup..."
bash scripts/runpod_setup.sh

# Start training
echo "🏋️ Starting training..."
cd /workspace/FusterCluck
bash start_training.sh
