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
        this.onBrushCreated = null;
        this.onBrushListReceived = null;
        this.onBrushLoaded = null;
        this.onBrushDeleted = null;

        // State
        this.isDrawing = false;
        this.lastRenderTime = Date.now();
        this.fps = 0;
    }

    connect() {
        console.log(`[WebSocket] Attempting to connect to ${this.serverUrl}...`);

        try {
            this.ws = new WebSocket(this.serverUrl);

            this.ws.onopen = () => {
                console.log('[WebSocket] ✓ Connection opened successfully');
                this.isConnected = true;
            };

            this.ws.onmessage = (event) => {
                console.log('[WebSocket] Message received:', event.data.substring(0, 100));
                this.handleMessage(event);
            };

            this.ws.onerror = (error) => {
                console.error('[WebSocket] ✗ Connection error:', error);
                console.error('[WebSocket] Server URL:', this.serverUrl);
                console.error('[WebSocket] Make sure the server is running!');
                if (this.onError) {
                    this.onError('WebSocket connection failed. Is the server running?');
                }
            };

            this.ws.onclose = (event) => {
                console.log('[WebSocket] ✗ Connection closed');
                console.log('[WebSocket] Close code:', event.code, 'Reason:', event.reason);
                this.isConnected = false;
                if (this.onDisconnected) {
                    this.onDisconnected();
                }
            };
        } catch (error) {
            console.error('[WebSocket] ✗ Failed to create WebSocket:', error);
            if (this.onError) {
                this.onError('Failed to create WebSocket connection');
            }
        }
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
                    console.log('[Message] ✓ Session connected:', this.sessionId);
                    console.log('[Message] Config:', this.config);
                    if (this.onConnected) {
                        console.log('[Message] Calling onConnected callback');
                        this.onConnected(data);
                    }
                    break;

                case 'render_update':
                    console.log('[Message] Render update received');
                    this.handleRenderUpdate(data);
                    break;

                case 'stats':
                    console.log('[Message] Stats update:', data);
                    if (this.onStatsUpdate) {
                        this.onStatsUpdate(data);
                    }
                    break;

                case 'error':
                    console.error('[Message] ✗ Server error:', data.message);
                    if (this.onError) {
                        this.onError(data.message);
                    }
                    break;

                case 'brush_params_updated':
                    console.log('[Message] ✓ Brush params updated:', data.params);
                    break;

                case 'brush_created':
                    console.log('[Message] ✓ Brush created:', data.pattern);
                    if (this.onBrushCreated) {
                        this.onBrushCreated(data);
                    }
                    // Request a render to show the brush
                    this.requestRender();
                    break;

                case 'scene_cleared':
                    console.log('[Message] ✓ Scene cleared');
                    break;

                case 'feature_flags_updated':
                    console.log('[Message] ✓ Feature flags updated:', data.flags);
                    break;

                case 'debug_mode_updated':
                    console.log('[Message] ✓ Debug mode updated:', data.enabled);
                    break;

                case 'debug_options_updated':
                    console.log('[Message] ✓ Debug options updated:', data.options);
                    break;

                case 'conversion_progress':
                    console.log('[Message] Conversion progress:', data.progress + '%', data.status);
                    if (this.onConversionProgress) {
                        this.onConversionProgress(data);
                    }
                    break;

                case 'conversion_complete':
                    console.log('[Message] ✓ Conversion complete:', data.brush_name);
                    if (this.onConversionComplete) {
                        this.onConversionComplete(data);
                    }
                    break;

                case 'conversion_failed':
                    console.error('[Message] ✗ Conversion failed:', data.error);
                    if (this.onConversionFailed) {
                        this.onConversionFailed(data);
                    }
                    break;

                case 'brush_list':
                    console.log('[Message] ✓ Brush list received:', data.brushes?.length || 0, 'brushes');
                    if (this.onBrushListReceived) {
                        this.onBrushListReceived(data.brushes || []);
                    }
                    break;

                case 'brush_loaded':
                    console.log('[Message] ✓ Brush loaded:', data.brush_id);
                    if (this.onBrushLoaded) {
                        this.onBrushLoaded(data);
                    }
                    break;

                case 'brush_deleted':
                    console.log('[Message] ✓ Brush deleted:', data.brush_id);
                    if (this.onBrushDeleted) {
                        this.onBrushDeleted(data.brush_id);
                    }
                    // Request updated brush list
                    this.requestBrushList();
                    break;

                default:
                    console.warn('[Message] ⚠ Unknown message type:', msgType, data);
            }
        } catch (error) {
            console.error('[Message] ✗ Error parsing message:', error);
            console.error('[Message] Raw data:', event.data);
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

    // Alias for send() to support BrushUploadHandler
    sendMessage(data) {
        this.send(data);
    }

    // Painting actions
    startStroke(x, y) {
        console.log(`[Stroke] ✓ Start stroke at (${x.toFixed(1)}, ${y.toFixed(1)})`);
        this.isDrawing = true;
        this.send({
            type: 'stroke_start',
            x: x,
            y: y
        });
    }

    updateStroke(x, y) {
        if (!this.isDrawing) {
            console.warn('[Stroke] ⚠ Update stroke called but not drawing');
            return;
        }

        console.log(`[Stroke] Update stroke at (${x.toFixed(1)}, ${y.toFixed(1)})`);
        this.send({
            type: 'stroke_update',
            x: x,
            y: y
        });
    }

    endStroke() {
        console.log('[Stroke] ✓ End stroke');
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

    setFeatureFlags(flags) {
        this.send({
            type: 'set_feature_flags',
            ...flags
        });
    }

    setDebugMode(enabled) {
        this.send({
            type: 'set_debug_mode',
            enabled: enabled
        });
    }

    setDebugOptions(options) {
        this.send({
            type: 'set_debug_options',
            options: options
        });
    }

    // Brush Library Methods
    requestBrushList() {
        this.send({
            type: 'list_brushes'
        });
    }

    loadBrush(brushId) {
        console.log('[Brush] Loading brush:', brushId);
        this.send({
            type: 'load_brush',
            brush_id: brushId
        });
    }

    deleteBrush(brushId) {
        console.log('[Brush] Deleting brush:', brushId);
        this.send({
            type: 'delete_brush',
            brush_id: brushId
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
