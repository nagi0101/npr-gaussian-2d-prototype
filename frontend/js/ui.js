/**
 * UI Controller
 *
 * Handles UI interactions and updates
 */

class UIController {
    constructor(client, canvasController) {
        this.client = client;
        this.canvasController = canvasController;

        // UI elements
        this.connectionStatus = document.getElementById('connection-status');
        this.loadingIndicator = document.getElementById('loading-indicator');

        // Brush controls
        this.brushPattern = document.getElementById('brushPattern');
        this.brushSize = document.getElementById('brushSize');
        this.brushSizeValue = document.getElementById('brushSizeValue');
        this.brushSpacing = document.getElementById('brushSpacing');
        this.brushSpacingValue = document.getElementById('brushSpacingValue');
        this.brushOpacity = document.getElementById('brushOpacity');
        this.brushOpacityValue = document.getElementById('brushOpacityValue');
        this.brushColor = document.getElementById('brushColor');

        // Buttons
        this.createBrushBtn = document.getElementById('createBrushBtn');
        this.applyBrushBtn = document.getElementById('applyBrushBtn');
        this.clearSceneBtn = document.getElementById('clearSceneBtn');
        this.requestRenderBtn = document.getElementById('requestRenderBtn');

        // Phase 3 Features
        this.enableDeformation = document.getElementById('enableDeformation');
        this.deformationStrength = document.getElementById('deformationStrength');
        this.deformationStrengthValue = document.getElementById('deformationStrengthValue');
        this.enableInpainting = document.getElementById('enableInpainting');
        this.applyFeaturesBtn = document.getElementById('applyFeaturesBtn');

        // Debug controls
        this.enableDebugMode = document.getElementById('enableDebugMode');
        this.debugOptions = document.getElementById('debugOptions');
        this.showGaussianOrigins = document.getElementById('showGaussianOrigins');
        this.showBasisVectors = document.getElementById('showBasisVectors');
        this.showSplineFrames = document.getElementById('showSplineFrames');
        this.debugOpacity = document.getElementById('debugOpacity');
        this.debugOpacityValue = document.getElementById('debugOpacityValue');
        this.basisVectorLength = document.getElementById('basisVectorLength');
        this.basisVectorLengthValue = document.getElementById('basisVectorLengthValue');
        this.applyDebugBtn = document.getElementById('applyDebugBtn');

        // Stats
        this.statGaussians = document.getElementById('stat-gaussians');
        this.statStrokes = document.getElementById('stat-strokes');
        this.statFps = document.getElementById('stat-fps');

        // Setup event listeners
        this.setupEventListeners();

        // Start FPS update loop
        this.startFPSUpdate();
    }

    setupEventListeners() {
        // Brush size slider
        this.brushSize.addEventListener('input', (e) => {
            this.brushSizeValue.textContent = e.target.value;
        });

        // Brush spacing slider
        this.brushSpacing.addEventListener('input', (e) => {
            this.brushSpacingValue.textContent = e.target.value;
        });

        // Brush opacity slider
        this.brushOpacity.addEventListener('input', (e) => {
            this.brushOpacityValue.textContent = e.target.value;
        });

        // Deformation strength slider
        this.deformationStrength.addEventListener('input', (e) => {
            this.deformationStrengthValue.textContent = e.target.value;
        });

        // Create brush button
        this.createBrushBtn.addEventListener('click', () => {
            this.handleCreateBrush();
        });

        // Apply brush params button
        this.applyBrushBtn.addEventListener('click', () => {
            this.handleApplyBrushParams();
        });

        // Clear scene button
        this.clearSceneBtn.addEventListener('click', () => {
            this.handleClearScene();
        });

        // Request render button
        this.requestRenderBtn.addEventListener('click', () => {
            this.client.requestRender();
        });

        // Apply features button
        this.applyFeaturesBtn.addEventListener('click', () => {
            this.handleApplyFeatures();
        });

        // Debug mode checkbox
        this.enableDebugMode.addEventListener('change', (e) => {
            this.handleToggleDebugMode(e.target.checked);
        });

        // Debug opacity slider
        this.debugOpacity.addEventListener('input', (e) => {
            this.debugOpacityValue.textContent = e.target.value;
        });

        // Basis vector length slider
        this.basisVectorLength.addEventListener('input', (e) => {
            this.basisVectorLengthValue.textContent = e.target.value;
        });

        // Apply debug settings button
        this.applyDebugBtn.addEventListener('click', () => {
            this.handleApplyDebugSettings();
        });
    }

    handleCreateBrush() {
        const pattern = this.brushPattern.value;
        const numGaussians = pattern === 'grid' ? 25 : 20;
        const color = this.hexToRgb(this.brushColor.value);

        console.log('Creating brush:', pattern, color);

        this.client.createBrush(pattern, numGaussians, color);
    }

    handleApplyBrushParams() {
        const params = {
            spacing: parseFloat(this.brushSpacing.value),
            opacity: parseFloat(this.brushOpacity.value),
            color: this.hexToRgb(this.brushColor.value)
        };

        console.log('Applying brush params:', params);

        this.client.setBrushParams(params);
    }

    handleClearScene() {
        if (confirm('Clear entire scene?')) {
            this.client.clearScene();
        }
    }

    handleApplyFeatures() {
        const flags = {
            enable_deformation: this.enableDeformation.checked,
            deformation_strength: parseFloat(this.deformationStrength.value),
            enable_inpainting: this.enableInpainting.checked
        };

        console.log('Applying feature flags:', flags);
        this.client.setFeatureFlags(flags);
    }

    handleToggleDebugMode(enabled) {
        // Show/hide debug options
        if (enabled) {
            this.debugOptions.classList.remove('hidden');
        } else {
            this.debugOptions.classList.add('hidden');
        }

        // Send debug mode state to server
        this.client.setDebugMode(enabled);
        console.log('Debug mode:', enabled ? 'ON' : 'OFF');
    }

    handleApplyDebugSettings() {
        const options = {
            show_gaussian_origins: this.showGaussianOrigins.checked,
            show_basis_vectors: this.showBasisVectors.checked,
            show_spline_frames: this.showSplineFrames.checked,
            debug_opacity: parseFloat(this.debugOpacity.value),
            basis_vector_length: parseInt(this.basisVectorLength.value)
        };

        console.log('Applying debug options:', options);
        this.client.setDebugOptions(options);
    }

    hexToRgb(hex) {
        // Remove # if present
        hex = hex.replace('#', '');

        // Parse hex values
        const r = parseInt(hex.substring(0, 2), 16) / 255.0;
        const g = parseInt(hex.substring(2, 4), 16) / 255.0;
        const b = parseInt(hex.substring(4, 6), 16) / 255.0;

        return [r, g, b];
    }

    updateConnectionStatus(connected) {
        if (connected) {
            this.connectionStatus.textContent = 'Connected';
            this.connectionStatus.className = 'status-connected';
        } else {
            this.connectionStatus.textContent = 'Disconnected';
            this.connectionStatus.className = 'status-disconnected';
        }
    }

    updateStats(data) {
        if (data.num_gaussians !== undefined) {
            this.statGaussians.textContent = data.num_gaussians;
        }

        if (data.num_strokes !== undefined) {
            this.statStrokes.textContent = data.num_strokes;
        }
    }

    startFPSUpdate() {
        setInterval(() => {
            const fps = this.client.getFPS();
            this.statFps.textContent = fps;
        }, 500);
    }

    showLoading() {
        this.loadingIndicator.classList.remove('hidden');
    }

    hideLoading() {
        this.loadingIndicator.classList.add('hidden');
    }
}

// Initialize everything when page loads
window.addEventListener('DOMContentLoaded', () => {
    console.log('[App] ========================================');
    console.log('[App] Starting application initialization');
    console.log('[App] ========================================');

    // Initialize WebSocket client
    console.log('[App] Step 1: Initializing WebSocket client');
    client = initWebSocketClient();

    // Get canvas element
    console.log('[App] Step 2: Getting canvas element');
    const canvas = document.getElementById('paintCanvas');
    if (!canvas) {
        console.error('[App] ✗ Canvas element not found!');
        return;
    }
    console.log('[App] ✓ Canvas element found:', canvas.id);

    // Setup client callbacks
    console.log('[App] Step 3: Setting up WebSocket callbacks');
    client.onConnected = (data) => {
        console.log('[App] ========================================');
        console.log('[App] ✓ WebSocket connected to server');
        console.log('[App] Session ID:', data.session_id);
        console.log('[App] ========================================');

        uiController.updateConnectionStatus(true);

        // Initialize canvas controller
        console.log('[App] Step 4: Initializing canvas controller');
        canvasController = initCanvasController(canvas, client);
        console.log('[App] ✓ Canvas controller initialized:', canvasController);

        // Setup render callback
        console.log('[App] Step 5: Setting up render callback');
        client.onRenderUpdate = (image, width, height) => {
            console.log('[App] Render update received, size:', width, 'x', height);
            canvasController.renderImage(image);
        };

        // Setup stats callback
        client.onStatsUpdate = (data) => {
            console.log('[App] Stats update:', data);
            uiController.updateStats(data);
        };

        // Setup error callback
        client.onError = (message) => {
            console.error('[App] ✗ Error:', message);
            alert('Error: ' + message);
        };

        uiController.hideLoading();
        console.log('[App] ========================================');
        console.log('[App] ✓ Application ready for painting!');
        console.log('[App] ========================================');
    };

    client.onDisconnected = () => {
        console.warn('[App] ⚠ Disconnected from server');
        uiController.updateConnectionStatus(false);
        alert('Disconnected from server. Please refresh the page.');
    };

    // Initialize UI controller
    console.log('[App] Step 6: Initializing UI controller');
    const uiController = new UIController(client, null);

    uiController.showLoading();

    console.log('[App] ✓ Application initialization complete');
    console.log('[App] Waiting for WebSocket connection...');
});
