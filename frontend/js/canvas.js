/**
 * Canvas interaction handling
 *
 * Handles mouse events and rendering on HTML5 canvas
 */

class CanvasController {
    constructor(canvasElement, client) {
        this.canvas = canvasElement;
        this.ctx = this.canvas.getContext('2d');
        this.client = client;

        this.isDrawing = false;
        this.lastPos = null;

        // Setup event listeners
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this.handleMouseLeave(e));

        // Touch events (for mobile)
        this.canvas.addEventListener('touchstart', (e) => this.handleTouchStart(e));
        this.canvas.addEventListener('touchmove', (e) => this.handleTouchMove(e));
        this.canvas.addEventListener('touchend', (e) => this.handleTouchEnd(e));

        // Prevent context menu
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    getCanvasCoordinates(clientX, clientY) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    handleMouseDown(e) {
        const pos = this.getCanvasCoordinates(e.clientX, e.clientY);
        this.isDrawing = true;
        this.lastPos = pos;

        // Send to server
        this.client.startStroke(pos.x, pos.y);
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
                return;  // Too close, skip
            }
        }

        this.lastPos = pos;

        // Send to server
        this.client.updateStroke(pos.x, pos.y);
    }

    handleMouseUp(e) {
        if (!this.isDrawing) return;

        this.isDrawing = false;
        this.lastPos = null;

        // Send to server
        this.client.endStroke();
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
            this.handleMouseDown({ clientX: touch.clientX, clientY: touch.clientY });
        }
    }

    handleTouchMove(e) {
        e.preventDefault();
        if (e.touches.length > 0) {
            const touch = e.touches[0];
            this.handleMouseMove({ clientX: touch.clientX, clientY: touch.clientY });
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
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
        };

        img.onerror = (error) => {
            console.error('Error loading image:', error);
        };

        img.src = 'data:image/png;base64,' + base64Image;
    }

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Fill with white
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

// Global canvas controller
let canvasController = null;

function initCanvasController(canvasElement, client) {
    canvasController = new CanvasController(canvasElement, client);
    return canvasController;
}
