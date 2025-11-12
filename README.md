# 3DGS 2D Painting Prototype

**2D Gaussian Splatting Brush Painting System**

Interactive painting application using 3D Gaussian Splatting techniques adapted for 2D canvas, inspired by the paper ["Painting with 3D Gaussian Splat Brushes"](https://doi.org/10.1145/3721238.3730724).

## 🎨 Features

-   **Real-time Gaussian Splatting Rendering**: CPU-based 2D Gaussian renderer
-   **Brush-based Painting**: Stamp-based brush system with multiple patterns
-   **Spline-based Strokes**: Smooth stroke paths with cubic spline interpolation
-   **Non-rigid Deformation**: Brush stamps deform along stroke curvature
-   **Interactive WebUI**: Browser-based painting interface
-   **WebSocket Real-time**: Low-latency client-server communication

## 🏗️ Architecture

```
├── backend/                    # Python backend
│   ├── core/
│   │   ├── gaussian.py         # Gaussian2D representation
│   │   ├── renderer.py         # 2D Gaussian renderer
│   │   ├── brush.py            # BrushStamp & StrokePainter
│   │   ├── spline.py           # Cubic spline fitting
│   │   └── deformation.py      # Non-rigid deformation
│   ├── api/
│   │   └── websocket.py        # WebSocket handler
│   └── main.py                 # FastAPI application
├── frontend/                   # HTML5/JS frontend
│   ├── index.html
│   ├── js/
│   │   ├── websocket_client.js
│   │   ├── canvas.js
│   │   └── ui.js
│   └── css/
│       └── style.css
└── tests/                      # Unit tests
```

## 🚀 Installation

### Prerequisites

-   Python 3.10+
-   conda (recommended for environment management)
-   NVIDIA GPU with CUDA 12.2 (for remote server deployment)

### Setup

#### 1. Clone Repository

```bash
git clone <repository-url>
cd npr-gaussian-2d-prototype
```

#### 2. Use Deployment Script (Recommended)

The deployment script automatically handles PyTorch CUDA installation:

```bash
chmod +x deploy.sh
./deploy.sh
```

This will:

-   Create/update conda environment
-   **Automatically install PyTorch with CUDA 12.1 support via pip**
-   Install pip dependencies
-   Optionally install GPU rasterizer
-   Start the server

#### 3. Manual Installation (Alternative)

If you prefer manual setup:

**a) Create Conda Environment:**

```bash
conda env create -f environment.yml
conda activate gaussian-brush-2d
```

**b) Install PyTorch with CUDA:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**c) Verify CUDA:**

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should output: CUDA: True
```

**d) Install pip dependencies:**

```bash
pip install -r requirements.txt
```

**e) (Optional) Install GPU-accelerated Gaussian Rasterizer:**

```bash
pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
```

## 🖥️ Running the Application

### Local Development

```bash
# Start the server
python -m backend.main

# Or use uvicorn directly
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at:

-   **Web Interface**: http://localhost:8000
-   **WebSocket**: ws://localhost:8000/ws
-   **Health Check**: http://localhost:8000/health

### Remote Server Deployment

For deployment on a remote server (e.g., with RTX 3090):

```bash
# SSH into server
ssh user@server

# Navigate to project directory
cd /path/to/npr-gaussian-2d-prototype

# Activate conda environment
conda activate gaussian-brush-2d

# Run server
python -m backend.main

# Or use nohup for background execution
nohup python -m backend.main > server.log 2>&1 &
```

Access from local machine:

```
http://<server-ip>:8000
```

### Using the Deployment Script

```bash
# Make script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

## 🎮 Usage

### Web Interface

1. **Open Browser**: Navigate to http://localhost:8000
2. **Wait for Connection**: Status indicator should show "Connected"
3. **Paint**: Click and drag on canvas to paint strokes
4. **Adjust Brush**: Use control panel to modify brush parameters
5. **Create Patterns**: Select different brush patterns (circular, line, grid)
6. **Clear**: Click "Clear Scene" to reset canvas

### Brush Parameters

-   **Pattern**: Circular, Line, or Grid arrangement of Gaussians
-   **Size**: Scale factor for brush (0.5 - 3.0)
-   **Spacing**: Distance between stamps along stroke (0.05 - 0.5)
-   **Opacity**: Transparency of Gaussians (0.1 - 1.0)
-   **Color**: RGB color of brush

### Keyboard Shortcuts

(To be implemented in future versions)

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Test individual modules
python backend/core/gaussian.py
python backend/core/renderer.py
python backend/core/spline.py
python backend/core/brush.py
python backend/core/deformation.py
```

## 📊 Performance

**Current Implementation** (CPU-based):

-   Rendering: ~5-10 FPS with 1000 Gaussians
-   Max Gaussians: 100,000 (configurable)
-   Resolution: 1024x768 (configurable)

**Future GPU Implementation**:

-   Expected: 30-60 FPS with 100,000+ Gaussians
-   Real-time deformation and inpainting

## 🐛 Troubleshooting

### GPU Rasterizer Installation Failed

If GPU rasterizer installation fails, check:

**1. Verify PyTorch CUDA installation**:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

Expected output: `CUDA: True, Version: 12.1`

**2. If CUDA is not available**, reinstall PyTorch:

```bash
conda activate gaussian-brush-2d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. Retry GPU rasterizer installation**:

```bash
pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
```

### CPU-Only Mode

The application works without GPU rasterizer (CPU rendering):

-   **Performance**: ~5-10 FPS with 1000 Gaussians
-   **No changes needed**: Just skip GPU rasterizer installation
-   **Good for**: Testing, small scenes, development

### GPU Mode (Recommended for Production)

With GPU rasterizer:

-   **Performance**: 30-60 FPS with 100,000+ Gaussians
-   **Requirements**: NVIDIA GPU with CUDA 12.x
-   **Server specs**: RTX 3090, CUDA 12.2, Driver 535+

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'torch'` during rasterizer build

-   **Cause**: PyTorch not in build environment
-   **Solution**: Use `--no-build-isolation` flag (already in deploy.sh)

**Issue**: `TypeError: expected string or bytes-like object` during build

-   **Cause**: PyTorch CPU version installed (torch.version.cuda is None)
-   **Solution**: Reinstall PyTorch with CUDA support (see step 2 above)

**Issue**: Environment creation fails

-   **Cause**: Channel conflicts or package unavailability
-   **Solution**:
    ```bash
    conda clean --all
    conda env create -f environment.yml --force
    ```

## 🔧 Configuration

Edit `backend/config.py` to modify:

```python
class Config:
    HOST = "0.0.0.0"
    PORT = 8000
    RENDER_WIDTH = 1024
    RENDER_HEIGHT = 768
    BACKGROUND_COLOR = (1.0, 1.0, 1.0)  # White
    WORLD_MIN = (-2.0, -2.0)
    WORLD_MAX = (2.0, 2.0)
    DEFAULT_BRUSH_SPACING = 0.1
    MAX_GAUSSIANS = 100000
```

## 📚 Technical Details

### Gaussian Representation

Each Gaussian is defined by:

-   **Position**: (x, y, 0) - constrained to z=0 plane
-   **Scale**: (sx, sy, sz_min) - z scale minimized for 2D
-   **Rotation**: Quaternion (x, y, z, w)
-   **Opacity**: Float [0, 1]
-   **Color**: RGB [0, 1]

### Rendering Algorithm

1. **Sort Gaussians**: Depth-based sorting (painter's algorithm)
2. **Compute Covariance**: 2D projection of 3D covariance
3. **Rasterize**: Per-pixel Gaussian evaluation
4. **Alpha Blending**: Front-to-back compositing

### Stroke Algorithm

1. **Input Points**: Mouse/touch coordinates
2. **Spline Fitting**: Cubic spline interpolation
3. **Arc-length Parameterization**: Uniform stamp spacing
4. **Stamp Placement**: Rigid transformation to spline frame
5. **Deformation**: Per-Gaussian warping along curvature
6. **(Future) Inpainting**: Diffusion-based seam removal

## 🛣️ Roadmap

### Phase 1: MVP (Current)

-   ✅ Basic Gaussian representation
-   ✅ 2D renderer (CPU)
-   ✅ Brush system with patterns
-   ✅ Spline-based strokes
-   ✅ WebSocket real-time communication
-   ✅ Interactive WebUI

### Phase 2: Advanced Features

-   ⬜ GPU-accelerated rendering (PyTorch/CUDA)
-   ⬜ Non-rigid deformation (implemented but not integrated)
-   ⬜ Diffusion inpainting for seamless strokes
-   ⬜ Multi-stroke blending
-   ⬜ Brush library system

### Phase 3: Production

-   ⬜ Undo/Redo system
-   ⬜ Save/Load scenes
-   ⬜ Export to images/videos
-   ⬜ Performance optimizations
-   ⬜ Mobile support

## 📖 References

1. **Painting with 3D Gaussian Splat Brushes**
   Pandey et al., SIGGRAPH 2025
   https://splatpainting.github.io

2. **3D Gaussian Splatting for Real-Time Radiance Field Rendering**
   Kerbl et al., SIGGRAPH 2023
   https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

-   Paper authors: Karran Pandey, Anita Hu, et al.
-   3DGS authors: Bernhard Kerbl, et al.
-   NVIDIA for computational resources

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Note**: This is a research prototype. The current implementation is CPU-based for rapid development. GPU acceleration is planned for production use.
