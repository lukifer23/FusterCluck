#!/bin/bash
# Cloud instance setup script for FusterCluck training

set -e

echo "Setting up FusterCluck cloud training environment..."

if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
fi

# Update system
$SUDO apt update && $SUDO apt upgrade -y

# Install system dependencies
$SUDO apt install -y \
    git \
    curl \
    wget \
    unzip \
    htop \
    tmux \
    vim \
    rsync \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv

# Install CUDA toolkit (if not already installed)
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA toolkit..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    $SUDO dpkg -i cuda-keyring_1.0-1_all.deb
    $SUDO apt update
    $SUDO apt install -y cuda-toolkit-12-1
fi

# Clone repository
if [ ! -d "FusterCluck" ]; then
    git clone https://github.com/lukifer23/FusterCluck.git
fi

cd FusterCluck

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -e .

# Install additional cloud dependencies
pip install \
    wandb[offline] \
    tensorboard \
    accelerate \
    datasets \
    transformers \
    webdataset \
    sentencepiece \
    tqdm \
    rich

# Create necessary directories
mkdir -p data/raw/cloud
mkdir -p data/processed/cloud
mkdir -p data/tokenized/cloud
mkdir -p artifacts/checkpoints/cloud
mkdir -p logs

# Set up WandB (offline mode)
export WANDB_MODE=offline
export WANDB_PROJECT=fustercluck-cloud

echo "Cloud environment setup complete!"
echo "Run 'source .venv/bin/activate' to activate the environment"
