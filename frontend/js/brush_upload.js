/**
 * Brush Upload Handler
 *
 * Handles image upload and conversion to 3DGS brushes
 */

class BrushUploadHandler {
    constructor(websocketClient) {
        this.client = websocketClient;
        this.currentUploadId = null;

        // UI elements
        this.fileInput = document.getElementById('brushImageUpload');
        this.uploadPreview = document.getElementById('uploadPreview');
        this.previewImage = document.getElementById('previewImage');
        this.previewFilename = document.getElementById('previewFilename');
        this.previewDimensions = document.getElementById('previewDimensions');
        this.depthProfile = document.getElementById('depthProfile');
        this.depthScale = document.getElementById('depthScale');
        this.depthScaleValue = document.getElementById('depthScaleValue');
        this.convertBtn = document.getElementById('convertBrushBtn');
        this.progressDiv = document.getElementById('conversionProgress');
        this.progressBar = this.progressDiv.querySelector('.progress-bar');
        this.progressText = this.progressDiv.querySelector('.progress-text');

        this.setupEventListeners();
    }

    setupEventListeners() {
        // File input change
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Depth scale slider
        this.depthScale.addEventListener('input', (e) => {
            this.depthScaleValue.textContent = e.target.value;
        });

        // Convert button
        this.convertBtn.addEventListener('click', () => this.handleConversion());

        // Drag and drop
        const dropZone = this.uploadPreview;

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFile(files[0]);
            }
        });

        // Listen for conversion updates from WebSocket
        this.client.onConversionProgress = (data) => this.updateProgress(data);
        this.client.onConversionComplete = (data) => this.handleConversionComplete(data);
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        // Validate file type
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
        if (!validTypes.includes(file.type)) {
            alert('Please select a valid image file (PNG, JPG, or BMP)');
            return;
        }

        // Validate file size (5MB max)
        const maxSize = 5 * 1024 * 1024;
        if (file.size > maxSize) {
            alert('File too large. Maximum size is 5MB');
            return;
        }

        // Show preview
        this.showPreview(file);

        // Upload file
        this.uploadFile(file);
    }

    showPreview(file) {
        const reader = new FileReader();

        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.previewFilename.textContent = file.name;

            // Get image dimensions
            const img = new Image();
            img.onload = () => {
                this.previewDimensions.textContent = `${img.width} × ${img.height}`;

                // Warn if image is too large
                if (img.width > 2048 || img.height > 2048) {
                    this.previewDimensions.textContent += ' (will be resized)';
                }
            };
            img.src = e.target.result;
        };

        reader.readAsDataURL(file);

        // Show preview section
        this.uploadPreview.classList.remove('hidden');
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('name', file.name.replace(/\.[^/.]+$/, ''));  // Remove extension
        formData.append('profile', this.depthProfile.value);

        try {
            this.convertBtn.disabled = true;
            this.convertBtn.textContent = 'Uploading...';

            const response = await fetch('/api/upload-brush-image', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.currentUploadId = result.upload_id;
                console.log('Upload successful:', result.upload_id);

                // Enable conversion button
                this.convertBtn.disabled = false;
                this.convertBtn.textContent = 'Convert to 3DGS Brush';
            } else {
                throw new Error(result.error || 'Upload failed');
            }

        } catch (error) {
            console.error('Upload error:', error);
            alert(`Upload failed: ${error.message}`);

            this.convertBtn.disabled = true;
            this.convertBtn.textContent = 'Upload Failed';
        }
    }

    handleConversion() {
        if (!this.currentUploadId) {
            alert('No image uploaded');
            return;
        }

        // Show progress
        this.progressDiv.classList.remove('hidden');
        this.progressBar.style.width = '0%';
        this.progressText.textContent = 'Starting conversion...';

        // Disable button
        this.convertBtn.disabled = true;

        // Send conversion request via WebSocket
        this.client.sendMessage({
            type: 'convert_brush_from_image',
            upload_id: this.currentUploadId,
            depth_profile: this.depthProfile.value,
            depth_scale: parseFloat(this.depthScale.value)
        });
    }

    updateProgress(data) {
        const progress = data.progress || 0;
        const status = data.status || 'Processing...';

        this.progressBar.style.width = `${progress}%`;
        this.progressText.textContent = status;

        if (progress >= 100) {
            this.progressText.textContent = 'Conversion complete!';
        }
    }

    handleConversionComplete(data) {
        console.log('Conversion complete:', data);

        // Hide progress after a moment
        setTimeout(() => {
            this.progressDiv.classList.add('hidden');

            // Reset UI
            this.convertBtn.disabled = false;
            this.convertBtn.textContent = 'Convert Another';

            // Clear current upload
            this.currentUploadId = null;
            this.fileInput.value = '';

            // Show success message
            this.progressText.textContent = `✓ Created brush with ${data.gaussian_count} Gaussians`;

        }, 2000);

        // The brush should now be available in the brush selector
        // You might want to refresh the brush list or select the new brush
        if (data.brush_name) {
            console.log(`New brush available: ${data.brush_name}`);
            // TODO: Update brush selector UI
        }
    }
}

// Export for use in main app
window.BrushUploadHandler = BrushUploadHandler;