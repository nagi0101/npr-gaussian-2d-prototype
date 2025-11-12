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
echo "======================================"
echo "  Ensuring PyTorch CUDA Installation"
echo "======================================"
echo "Checking current PyTorch installation..."
python -c "import torch; print(f'Current: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not yet installed"

echo ""
echo "Installing/Updating PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
else:
    print('ERROR: CUDA still not available!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to install PyTorch with CUDA support"
    echo ""
    echo "Manual installation required:"
    echo "  conda activate ${ENV_NAME}"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    echo ""
    read -p "Continue with CPU-only mode? (y/N): " continue_cpu
    if [[ ! $continue_cpu =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Installing pip dependencies..."
pip install -r requirements.txt

echo ""
echo "======================================"
echo "  PyTorch CUDA Check"
echo "======================================"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  CUDA not available - CPU mode only')
" || {
    echo "❌ Error: PyTorch not installed properly"
    exit 1
}

echo ""
echo "======================================"
echo "  Optional: gsplat (GPU Rasterizer)"
echo "======================================"
echo "gsplat: High-performance Gaussian splatting library"
echo "  - 50-100 FPS with 1000 Gaussians"
echo "  - Native 2D orthographic support"
echo "  - 4x less memory than alternatives"
echo ""
read -p "Install gsplat for GPU acceleration? (Y/n): " install_gpu

# Default to Yes if empty
install_gpu=${install_gpu:-Y}

if [[ $install_gpu =~ ^[Yy]$ ]]; then
    # Check if PyTorch has CUDA support
    HAS_CUDA=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)

    if [[ "$HAS_CUDA" == "True" ]]; then
        echo "✓ PyTorch CUDA detected"
        echo "Installing gsplat..."
        pip install gsplat || {
            echo "⚠️  gsplat installation failed"
            echo "Continuing with CPU-only mode..."
            echo ""
            echo "Manual installation:"
            echo "  pip install gsplat"
        }
    else
        echo "⚠️  PyTorch CUDA not available"
        echo ""
        echo "To enable GPU acceleration, reinstall PyTorch with CUDA:"
        echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        echo ""
        echo "Continuing with CPU-only mode..."
    fi
else
    echo "Skipping gsplat installation (CPU-only mode)"
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
