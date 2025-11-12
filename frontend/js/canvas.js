/**
 * Canvas interaction handling
 *
 * Handles mouse events and rendering on HTML5 canvas
 */

class CanvasController {
    constructor(canvasElement, client) {
        console.log("[Canvas] Initializing CanvasController");
        this.canvas = canvasElement;
        this.ctx = this.canvas.getContext("2d");
        this.client = client;

        this.isDrawing = false;
        this.lastPos = null;

        console.log(
            "[Canvas] Canvas size:",
            this.canvas.width,
            "x",
            this.canvas.height
        );
        console.log("[Canvas] Client connected:", this.client.isConnected);

        // Setup event listeners
        this.setupEventListeners();
        console.log("[Canvas] ✓ CanvasController initialized");
    }

    setupEventListeners() {
        console.log("[Canvas] Setting up event listeners");

        // Mouse events
        this.canvas.addEventListener("mousedown", (e) =>
            this.handleMouseDown(e)
        );
        this.canvas.addEventListener("mousemove", (e) =>
            this.handleMouseMove(e)
        );
        this.canvas.addEventListener("mouseup", (e) => this.handleMouseUp(e));
        this.canvas.addEventListener("mouseleave", (e) =>
            this.handleMouseLeave(e)
        );

        // Touch events (for mobile)
        this.canvas.addEventListener("touchstart", (e) =>
            this.handleTouchStart(e)
        );
        this.canvas.addEventListener("touchmove", (e) =>
            this.handleTouchMove(e)
        );
        this.canvas.addEventListener("touchend", (e) => this.handleTouchEnd(e));

        // Prevent context menu
        this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

        console.log("[Canvas] ✓ All event listeners registered");
    }

    getCanvasCoordinates(clientX, clientY) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY,
        };
    }

    handleMouseDown(e) {
        console.log("[Canvas] ✓ Mouse down event");
        const pos = this.getCanvasCoordinates(e.clientX, e.clientY);
        console.log(
            "[Canvas] Canvas coordinates:",
            pos.x.toFixed(1),
            pos.y.toFixed(1)
        );
        this.isDrawing = true;
        this.lastPos = pos;

        // Send to server
        if (this.client) {
            this.client.startStroke(pos.x, pos.y);
        } else {
            console.error("[Canvas] ✗ Client is null! Cannot start stroke");
        }
    }

    handleMouseMove(e) {
        if (!this.isDrawing) return;

        const pos = this.getCanvasCoordinates(e.clientX, e.clientY);

        // Throttle updates (every few pixels)
        if (this.lastPos) {
            const dx = pos.x - this.lastPos.x;
            const dy = pos.y - this.lastPos.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < 2) {
                return; // Too close, skip
            }
        }

        this.lastPos = pos;

        // Send to server
        if (this.client) {
            this.client.updateStroke(pos.x, pos.y);
        } else {
            console.error("[Canvas] ✗ Client is null! Cannot update stroke");
        }
    }

    handleMouseUp(e) {
        if (!this.isDrawing) return;

        console.log("[Canvas] ✓ Mouse up event");
        this.isDrawing = false;
        this.lastPos = null;

        // Send to server
        if (this.client) {
            this.client.endStroke();
        } else {
            console.error("[Canvas] ✗ Client is null! Cannot end stroke");
        }
    }

    handleMouseLeave(e) {
        if (this.isDrawing) {
            this.handleMouseUp(e);
        }
    }

    // Touch event handlers
    handleTouchStart(e) {
        e.preventDefault();
        if (e.touches.length > 0) {
            const touch = e.touches[0];
            this.handleMouseDown({
                clientX: touch.clientX,
                clientY: touch.clientY,
            });
        }
    }

    handleTouchMove(e) {
        e.preventDefault();
        if (e.touches.length > 0) {
            const touch = e.touches[0];
            this.handleMouseMove({
                clientX: touch.clientX,
                clientY: touch.clientY,
            });
        }
    }

    handleTouchEnd(e) {
        e.preventDefault();
        this.handleMouseUp(e);
    }

    // Render image from base64
    renderImage(base64Image) {
        const img = new Image();

        img.onload = () => {
            // Clear canvas
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

            // Draw image
            this.ctx.drawImage(
                img,
                0,
                0,
                this.canvas.width,
                this.canvas.height
            );
        };

        img.onerror = (error) => {
            console.error("Error loading image:", error);
        };

        // Use JPEG for 4-6x faster encoding/transmission
        img.src = "data:image/jpeg;base64," + base64Image;
    }

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Fill with white
        this.ctx.fillStyle = "white";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

// Global canvas controller
let canvasController = null;

function initCanvasController(canvasElement, client) {
    canvasController = new CanvasController(canvasElement, client);
    return canvasController;
}
