/**
 * Brush Preview System
 *
 * Displays a visual preview of the current brush
 */

class BrushPreview {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.currentBrushData = null;
        this.setupPreviewCanvas();
    }

    setupPreviewCanvas() {
        // Create canvas element
        this.canvas = document.createElement('canvas');
        this.canvas.width = 150;
        this.canvas.height = 150;
        this.canvas.style.border = '1px solid #ddd';
        this.canvas.style.borderRadius = '8px';
        this.canvas.style.backgroundColor = '#f9f9f9';
        this.canvas.style.margin = '10px auto';
        this.canvas.style.display = 'block';
        this.canvas.id = 'brushPreviewCanvas';

        this.ctx = this.canvas.getContext('2d');

        // Find or create preview container
        let container = document.getElementById('brushPreviewContainer');
        if (!container) {
            // Create container and add to UI
            container = document.createElement('div');
            container.id = 'brushPreviewContainer';
            container.style.padding = '10px';
            container.style.backgroundColor = 'white';
            container.style.borderRadius = '8px';
            container.style.marginTop = '10px';
            container.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';

            // Add title
            const title = document.createElement('h4');
            title.textContent = 'Current Brush Preview';
            title.style.margin = '0 0 10px 0';
            title.style.textAlign = 'center';
            title.style.fontSize = '14px';
            title.style.color = '#333';
            container.appendChild(title);

            // Add canvas
            container.appendChild(this.canvas);

            // Add info div
            const infoDiv = document.createElement('div');
            infoDiv.id = 'brushPreviewInfo';
            infoDiv.style.textAlign = 'center';
            infoDiv.style.fontSize = '12px';
            infoDiv.style.color = '#666';
            infoDiv.style.marginTop = '8px';
            container.appendChild(infoDiv);

            // Try to insert into the UI
            const uploadSection = document.querySelector('#uploadPreview')?.parentElement;
            if (uploadSection) {
                uploadSection.appendChild(container);
            } else {
                // Fallback: add to control panel
                const controlPanel = document.getElementById('control-panel');
                if (controlPanel) {
                    controlPanel.appendChild(container);
                }
            }
        }

        this.container = container;
    }

    updatePreview(brushData) {
        if (!brushData || !brushData.gaussians || brushData.gaussians.length === 0) {
            this.clearPreview();
            return;
        }

        this.currentBrushData = brushData;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw background
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Visualize Gaussians as colored dots
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const scale = 40; // Scale factor to fit brush in preview

        // Sort by Z for proper depth rendering
        const sortedGaussians = [...brushData.gaussians].sort((a, b) => {
            return (a.position?.[2] || 0) - (b.position?.[2] || 0);
        });

        // Draw each Gaussian
        sortedGaussians.forEach(gaussian => {
            if (!gaussian.position || !gaussian.color) return;

            // Project to 2D (simple orthographic projection)
            const x = centerX + gaussian.position[0] * scale;
            const y = centerY - gaussian.position[1] * scale; // Flip Y

            // Get color and opacity
            const r = Math.floor((gaussian.color[0] || 0.5) * 255);
            const g = Math.floor((gaussian.color[1] || 0.5) * 255);
            const b = Math.floor((gaussian.color[2] || 0.5) * 255);
            const opacity = gaussian.opacity || 0.8;

            // Draw as circle
            const radius = Math.max(1, (gaussian.scale?.[0] || 0.05) * scale * 0.5);

            this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${opacity})`;
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius, 0, Math.PI * 2);
            this.ctx.fill();
        });

        // Update info
        this.updateInfo(brushData);

        // Show container
        if (this.container) {
            this.container.style.display = 'block';
        }
    }

    updateInfo(brushData) {
        const infoDiv = document.getElementById('brushPreviewInfo');
        if (!infoDiv) return;

        const gaussianCount = brushData.gaussians?.length || 0;
        const brushName = brushData.metadata?.name || 'Converted Brush';
        const brushSize = brushData.size?.toFixed(3) || 'N/A';

        infoDiv.innerHTML = `
            <div><strong>${brushName}</strong></div>
            <div>${gaussianCount} Gaussians | Size: ${brushSize}</div>
        `;
    }

    clearPreview() {
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.fillStyle = '#f9f9f9';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

            // Draw placeholder text
            this.ctx.fillStyle = '#ccc';
            this.ctx.font = '12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('No brush loaded', this.canvas.width / 2, this.canvas.height / 2);
        }

        const infoDiv = document.getElementById('brushPreviewInfo');
        if (infoDiv) {
            infoDiv.innerHTML = '<div style="color: #999;">No brush selected</div>';
        }
    }

    hide() {
        if (this.container) {
            this.container.style.display = 'none';
        }
    }

    show() {
        if (this.container) {
            this.container.style.display = 'block';
        }
    }
}

// Export for use
window.BrushPreview = BrushPreview;