/**
 * WebSocket Client for 3DGS Painting
 *
 * Handles connection and communication with backend server
 */

class GaussianPaintClient {
    constructor(serverUrl) {
        this.serverUrl = serverUrl;
        this.ws = null;
        this.isConnected = false;
        this.sessionId = null;
        this.config = null;

        // Callbacks
        this.onConnected = null;
        this.onDisconnected = null;
        this.onRenderUpdate = null;
        this.onStatsUpdate = null;
        this.onError = null;

        // State
        this.isDrawing = false;
        this.lastRenderTime = Date.now();
        this.fps = 0;
    }

    connect() {
        console.log(`Connecting to ${this.serverUrl}...`);

        this.ws = new WebSocket(this.serverUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
        };

        this.ws.onmessage = (event) => {
            this.handleMessage(event);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (this.onError) {
                this.onError('Connection error');
            }
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            if (this.onDisconnected) {
                this.onDisconnected();
            }
        };
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            const msgType = data.type;

            switch (msgType) {
                case 'connected':
                    this.sessionId = data.session_id;
                    this.config = data.config;
                    console.log('Session connected:', this.sessionId);
                    if (this.onConnected) {
                        this.onConnected(data);
                    }
                    break;

                case 'render_update':
                    this.handleRenderUpdate(data);
                    break;

                case 'stats':
                    if (this.onStatsUpdate) {
                        this.onStatsUpdate(data);
                    }
                    break;

                case 'error':
                    console.error('Server error:', data.message);
                    if (this.onError) {
                        this.onError(data.message);
                    }
                    break;

                case 'brush_params_updated':
                    console.log('Brush params updated:', data.params);
                    break;

                case 'brush_created':
                    console.log('Brush created:', data.pattern);
                    break;

                case 'scene_cleared':
                    console.log('Scene cleared');
                    break;

                default:
                    console.log('Unknown message type:', msgType, data);
            }
        } catch (error) {
            console.error('Error parsing message:', error);
        }
    }

    handleRenderUpdate(data) {
        // Update FPS
        const now = Date.now();
        const dt = (now - this.lastRenderTime) / 1000.0;
        this.fps = dt > 0 ? (1.0 / dt) : 0;
        this.lastRenderTime = now;

        // Call render callback
        if (this.onRenderUpdate) {
            this.onRenderUpdate(data.image, data.width, data.height);
        }
    }

    send(data) {
        if (!this.isConnected || !this.ws) {
            console.warn('Cannot send: not connected');
            return;
        }

        try {
            this.ws.send(JSON.stringify(data));
        } catch (error) {
            console.error('Error sending message:', error);
        }
    }

    // Painting actions
    startStroke(x, y) {
        this.isDrawing = true;
        this.send({
            type: 'stroke_start',
            x: x,
            y: y
        });
    }

    updateStroke(x, y) {
        if (!this.isDrawing) return;

        this.send({
            type: 'stroke_update',
            x: x,
            y: y
        });
    }

    endStroke() {
        this.isDrawing = false;
        this.send({
            type: 'stroke_end'
        });
    }

    clearScene() {
        this.send({
            type: 'clear_scene'
        });
    }

    requestRender() {
        this.send({
            type: 'request_render'
        });
    }

    setBrushParams(params) {
        this.send({
            type: 'set_brush_params',
            ...params
        });
    }

    createBrush(pattern, numGaussians, color) {
        this.send({
            type: 'create_brush',
            pattern: pattern,
            num_gaussians: numGaussians,
            color: color
        });
    }

    getFPS() {
        return Math.round(this.fps);
    }
}

// Create global client instance
// Will be initialized when page loads
let client = null;

function initWebSocketClient() {
    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:8000';
    const wsUrl = `${protocol}//${host}/ws`;

    console.log('WebSocket URL:', wsUrl);

    client = new GaussianPaintClient(wsUrl);
    client.connect();

    return client;
}
