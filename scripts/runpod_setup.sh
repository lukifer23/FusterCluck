#!/bin/bash
# RunPod-specific setup script for FusterCluck training

set -e

echo "🚀 Setting up FusterCluck on RunPod..."

# Check if we're on RunPod
if [ ! -f "/runpod/runpod.sock" ]; then
    echo "⚠️  This script is optimized for RunPod. Proceeding anyway..."
fi

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y \
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
    python3-venv \
    nvidia-utils-535

# Check GPU
echo "🎮 Checking GPU..."
nvidia-smi

# Clone repository to /workspace (persistent storage)
echo "📥 Cloning FusterCluck repository..."
cd /workspace
if [ ! -d "FusterCluck" ]; then
    git clone https://github.com/lukifer23/FusterCluck.git
fi

cd FusterCluck

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
echo "📚 Installing project dependencies..."
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
    rich \
    psutil

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /workspace/FusterCluck/data/raw/cloud
mkdir -p /workspace/FusterCluck/data/processed/cloud
mkdir -p /workspace/FusterCluck/data/tokenized/cloud
mkdir -p /workspace/FusterCluck/artifacts/checkpoints/cloud
mkdir -p /workspace/FusterCluck/logs

# Set up environment variables
echo "⚙️  Setting up environment..."
export WANDB_MODE=offline
export WANDB_PROJECT=fustercluck-runpod
export CUDA_VISIBLE_DEVICES=0

# Create a startup script
echo "📝 Creating startup script..."
cat > /workspace/FusterCluck/start_training.sh << 'EOF'
#!/bin/bash
cd /workspace/FusterCluck
source .venv/bin/activate
export WANDB_MODE=offline
export WANDB_PROJECT=fustercluck-runpod

echo "🚀 Starting FusterCluck training..."
echo "📊 GPU Info:"
nvidia-smi

echo "💾 Disk Usage:"
df -h /workspace

echo "🏋️  Starting training pipeline..."
bash scripts/quick_start_cloud.sh
EOF

chmod +x /workspace/FusterCluck/start_training.sh

# Set up Jupyter for monitoring (optional)
echo "📊 Setting up Jupyter for monitoring..."
pip install jupyter
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py

echo "✅ RunPod setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Run: cd /workspace/FusterCluck && bash start_training.sh"
echo "2. Or run individual stages:"
echo "   - python scripts/run_cloud_training.py --stage 1"
echo "   - python scripts/run_cloud_training.py --stage 2"
echo ""
echo "📊 Monitor progress:"
echo "- Check logs: tail -f /workspace/FusterCluck/logs/cloud_training.log"
echo "- Monitor GPU: watch nvidia-smi"
echo "- Check disk: df -h /workspace"
