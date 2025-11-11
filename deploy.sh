#!/bin/bash
# Deployment script for 3DGS 2D Painting Backend
# For use on remote server with conda and CUDA 12.2

set -e  # Exit on error

echo "======================================"
echo "  3DGS 2D Painting - Deployment"
echo "======================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found"
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

echo "✓ conda found"

# Environment name
ENV_NAME="gaussian-brush-2d"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "✓ Environment '${ENV_NAME}' exists"

    # Ask if user wants to recreate
    read -p "Do you want to recreate the environment? (y/N): " recreate
    if [[ $recreate =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
        echo "Creating new environment..."
        conda env create -f environment.yml
    fi
else
    echo "Creating environment '${ENV_NAME}'..."
    conda env create -f environment.yml
fi

echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo ""
echo "Installing pip dependencies..."
pip install -r requirements.txt

echo ""
echo "======================================"
echo "  Optional: GPU Rasterizer"
echo "======================================"
read -p "Install diff-gaussian-rasterization (GPU)? (y/N): " install_gpu

if [[ $install_gpu =~ ^[Yy]$ ]]; then
    echo "Installing GPU rasterizer..."
    pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git || {
        echo "⚠️  GPU rasterizer installation failed (may require CUDA toolkit)"
        echo "Continuing without GPU acceleration..."
    }
fi

echo ""
echo "======================================"
echo "  Server Configuration"
echo "======================================"

# Ask for host and port
read -p "Host (default: 0.0.0.0): " host
host=${host:-0.0.0.0}

read -p "Port (default: 8000): " port
port=${port:-8000}

echo ""
echo "======================================"
echo "  Starting Server"
echo "======================================"
echo ""
echo "Server will start at: http://${host}:${port}"
echo "WebSocket endpoint: ws://${host}:${port}/ws"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run server
python -m backend.main

echo ""
echo "Server stopped."
