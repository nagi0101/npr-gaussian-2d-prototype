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
    console.log('Initializing application...');

    // Initialize WebSocket client
    client = initWebSocketClient();

    // Get canvas element
    const canvas = document.getElementById('paintCanvas');

    // Setup client callbacks
    client.onConnected = (data) => {
        console.log('Connected to server:', data);
        uiController.updateConnectionStatus(true);

        // Initialize canvas controller
        canvasController = initCanvasController(canvas, client);

        // Setup render callback
        client.onRenderUpdate = (image, width, height) => {
            canvasController.renderImage(image);
        };

        // Setup stats callback
        client.onStatsUpdate = (data) => {
            uiController.updateStats(data);
        };

        // Setup error callback
        client.onError = (message) => {
            alert('Error: ' + message);
        };

        uiController.hideLoading();
    };

    client.onDisconnected = () => {
        uiController.updateConnectionStatus(false);
        alert('Disconnected from server. Please refresh the page.');
    };

    // Initialize UI controller
    const uiController = new UIController(client, null);

    uiController.showLoading();

    console.log('Application initialized');
});
