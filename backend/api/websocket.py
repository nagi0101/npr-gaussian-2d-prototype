"""WebSocket handler for real-time painting"""

import json
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Optional
import asyncio
import traceback

from backend.core.gaussian import Gaussian2D
from backend.core.brush import BrushStamp, StrokePainter
from backend.core.renderer import create_renderer
from backend.utils.helpers import numpy_to_base64_jpeg
from backend.config import config


class PaintingSession:
    """
    WebSocket session for painting

    각 클라이언트 연결마다 하나의 세션 인스턴스 생성
    """

    def __init__(self, websocket: WebSocket, session_id: str):
        """
        Initialize painting session

        Args:
            websocket: WebSocket connection
            session_id: Unique session identifier
        """
        print(f"[Session {session_id}] Initializing painting session")
        self.ws = websocket
        self.session_id = session_id

        # Renderer (GPU-accelerated with automatic fallback)
        print(f"[Session {session_id}] Creating renderer {config.RENDER_WIDTH}x{config.RENDER_HEIGHT}")
        self.renderer = create_renderer(
            width=config.RENDER_WIDTH,
            height=config.RENDER_HEIGHT,
            background_color=np.array(config.BACKGROUND_COLOR),
            prefer_gpu=True,  # Try GPU first, fallback to CPU
            debug_mode=config.DEBUG_MODE_ENABLED  # Enable debug mode from config
        )
        self.renderer.set_world_bounds(
            np.array(config.WORLD_MIN),
            np.array(config.WORLD_MAX)
        )

        # Initialize debug options from config
        if config.DEBUG_MODE_ENABLED:
            self.renderer.set_debug_options({
                'show_gaussian_origins': config.SHOW_GAUSSIAN_ORIGINS,
                'show_basis_vectors': config.SHOW_BASIS_VECTORS,
                'show_spline_frames': config.SHOW_SPLINE_FRAMES,
                'show_deformation_vectors': config.SHOW_DEFORMATION_VECTORS,
                'debug_opacity': config.DEBUG_OVERLAY_OPACITY,
                'basis_vector_length': config.BASIS_VECTOR_LENGTH
            })

        # Scene state
        self.scene_gaussians: list[Gaussian2D] = []
        self.current_painter: Optional[StrokePainter] = None
        self.brush: Optional[BrushStamp] = None

        # Phase 3 feature flags
        self.enable_deformation = config.ENABLE_DEFORMATION
        self.enable_inpainting = config.ENABLE_INPAINTING

        # Initialize default brush
        print(f"[Session {session_id}] Creating default brush")
        self._create_default_brush()
        print(f"[Session {session_id}] ✓ Session initialized successfully")

    def _create_default_brush(self):
        """Create default circular brush"""
        self.brush = BrushStamp()
        self.brush.create_circular_pattern(
            num_gaussians=15,
            radius=0.15,
            gaussian_scale=0.03,
            opacity=config.DEFAULT_GAUSSIAN_OPACITY,
            color=np.array([0.3, 0.3, 0.3])  # Dark gray
        )
        self.brush.spacing = config.DEFAULT_BRUSH_SPACING

        # Initialize painter
        self.current_painter = StrokePainter(self.brush, self.scene_gaussians)

    async def handle_message(self, data: dict):
        """
        Handle incoming WebSocket message

        Args:
            data: JSON message from client
        """
        try:
            msg_type = data.get('type')

            if msg_type == 'stroke_start':
                await self._handle_stroke_start(data)
            elif msg_type == 'stroke_update':
                await self._handle_stroke_update(data)
            elif msg_type == 'stroke_end':
                await self._handle_stroke_end(data)
            elif msg_type == 'clear_scene':
                await self._handle_clear_scene()
            elif msg_type == 'set_brush_params':
                await self._handle_set_brush_params(data)
            elif msg_type == 'create_brush':
                await self._handle_create_brush(data)
            elif msg_type == 'set_feature_flags':
                await self._handle_set_feature_flags(data)
            elif msg_type == 'set_debug_mode':
                await self._handle_set_debug_mode(data)
            elif msg_type == 'set_debug_options':
                await self._handle_set_debug_options(data)
            elif msg_type == 'convert_brush_from_image':
                await self._handle_convert_brush_from_image(data)
            elif msg_type == 'request_render':
                await self.send_render()
            # Brush library management
            elif msg_type == 'save_brush':
                await self._handle_save_brush(data)
            elif msg_type == 'load_brush':
                await self._handle_load_brush(data)
            elif msg_type == 'list_brushes':
                await self._handle_list_brushes()
            elif msg_type == 'delete_brush':
                await self._handle_delete_brush(data)
            elif msg_type == 'update_brush_metadata':
                await self._handle_update_brush_metadata(data)
            else:
                print(f"[Session {self.session_id}] ✗ Unknown message type: {msg_type}")
                await self.send_error(f"Unknown message type: {msg_type}")

        except Exception as e:
            error_msg = f"Error handling message: {str(e)}\n{traceback.format_exc()}"
            print(f"[Session {self.session_id}] ✗ ERROR: {error_msg}")
            await self.send_error(error_msg)

    async def _handle_stroke_start(self, data: dict):
        """Handle stroke start"""
        # Get position from client (pixel coordinates)
        pixel_x = data.get('x', 0)
        pixel_y = data.get('y', 0)
        print(f"[Session {self.session_id}] ✓ Stroke start at pixel ({pixel_x:.1f}, {pixel_y:.1f})")

        # Convert to world coordinates
        world_pos = self.renderer.pixel_to_world(np.array([pixel_x, pixel_y]))
        position_3d = np.array([world_pos[0], world_pos[1], 0.0])
        print(f"[Session {self.session_id}] World coordinates: ({world_pos[0]:.3f}, {world_pos[1]:.3f})")

        # Default normal (pointing up)
        normal = np.array([0, 0, 1], dtype=np.float32)

        # Start stroke with deformation flag
        self.current_painter.start_stroke(position_3d, normal, enable_deformation=self.enable_deformation)

        # Send render
        print(f"[Session {self.session_id}] Sending render update")
        await self.send_render()

    async def _handle_stroke_update(self, data: dict):
        """Handle stroke update (mouse move while drawing)"""
        # Get position
        pixel_x = data.get('x', 0)
        pixel_y = data.get('y', 0)

        # Convert to world coordinates
        world_pos = self.renderer.pixel_to_world(np.array([pixel_x, pixel_y]))
        position_3d = np.array([world_pos[0], world_pos[1], 0.0])

        # Default normal
        normal = np.array([0, 0, 1], dtype=np.float32)

        # Update stroke
        self.current_painter.update_stroke(position_3d, normal)

        # Send render
        await self.send_render()

    async def _handle_stroke_end(self, data: dict):
        """Handle stroke end"""
        # Pass feature flags to finish_stroke
        self.current_painter.finish_stroke(
            enable_deformation=self.enable_deformation,
            enable_inpainting=self.enable_inpainting
        )

        # Send final render
        await self.send_render()

        # Send stats
        await self.send_stats()

    async def _handle_clear_scene(self):
        """Clear entire scene"""
        self.scene_gaussians.clear()
        self.current_painter.clear_scene()

        await self.send_render()
        await self.send_message({
            'type': 'scene_cleared'
        })

    async def _handle_set_brush_params(self, data: dict):
        """Set brush parameters using the new parameter system"""
        # Collect parameters to apply
        params_to_apply = {}

        if 'spacing' in data:
            params_to_apply['spacing'] = float(data['spacing'])

        if 'size' in data:
            # Size is now a multiplier, not cumulative
            params_to_apply['size_multiplier'] = float(data['size'])

        if 'opacity' in data:
            # This is global opacity multiplier
            params_to_apply['global_opacity'] = float(data['opacity'])

        if 'color' in data:
            color = np.array(data['color'], dtype=np.float32)
            params_to_apply['color'] = color

        # Apply all parameters at once
        if params_to_apply:
            self.brush.apply_parameters(**params_to_apply)

        await self.send_message({
            'type': 'brush_params_updated',
            'params': {
                'spacing': self.brush.spacing,
                'size': self.brush.current_size_multiplier,
                'opacity': self.brush.current_global_opacity,
                'color': self.brush.current_color.tolist()
            }
        })

    async def _handle_create_brush(self, data: dict):
        """Create new brush with specified pattern"""
        pattern = data.get('pattern', 'circular')
        num_gaussians = data.get('num_gaussians', 20)
        color = np.array(data.get('color', [0.3, 0.3, 0.3]), dtype=np.float32)

        self.brush = BrushStamp()

        if pattern == 'circular':
            self.brush.create_circular_pattern(
                num_gaussians=num_gaussians,
                radius=0.15,
                gaussian_scale=0.03,
                opacity=0.8,
                color=color
            )
        elif pattern == 'line':
            self.brush.create_line_pattern(
                num_gaussians=num_gaussians,
                length=0.3,
                thickness=0.03,
                opacity=0.8,
                color=color
            )
        elif pattern == 'grid':
            grid_size = int(np.sqrt(num_gaussians))
            self.brush.create_grid_pattern(
                grid_size=grid_size,
                spacing=0.05,
                gaussian_scale=0.02,
                opacity=0.8,
                color=color
            )

        self.brush.spacing = config.DEFAULT_BRUSH_SPACING

        # Apply the requested color to the pattern
        self.brush.apply_parameters(color=color)

        # Recreate painter with new brush
        self.current_painter = StrokePainter(self.brush, self.scene_gaussians)

        # Auto-save programmatic brush to library
        from backend.core.brush_manager import get_brush_manager
        brush_manager = get_brush_manager()

        # Generate descriptive name based on pattern
        pattern_names = {
            'circular': 'Circular Brush',
            'line': 'Line Brush',
            'grid': 'Grid Brush'
        }
        brush_name = pattern_names.get(pattern, 'Custom Brush')

        # Save the brush
        brush_id = brush_manager.save_brush(
            self.brush,
            name=brush_name,
            brush_type='programmatic',
            source=pattern
        )

        await self.send_message({
            'type': 'brush_created',
            'pattern': pattern,
            'num_gaussians': len(self.brush.gaussians),
            'brush_id': brush_id,
            'brush_name': brush_name
        })

        # Send updated brush list
        brushes = brush_manager.list_brushes()
        await self.send_message({
            'type': 'brush_list',
            'brushes': brushes
        })

        # Send a render update to show the brush preview
        await self.send_render()

    async def _handle_set_feature_flags(self, data: dict):
        """Set Phase 3 feature flags"""
        if 'enable_deformation' in data:
            self.enable_deformation = bool(data['enable_deformation'])
            print(f"[Session {self.session_id}] ✓ Deformation: {'ON' if self.enable_deformation else 'OFF'}")

        if 'enable_inpainting' in data:
            self.enable_inpainting = bool(data['enable_inpainting'])
            print(f"[Session {self.session_id}] ✓ Inpainting: {'ON' if self.enable_inpainting else 'OFF'}")

        await self.send_message({
            'type': 'feature_flags_updated',
            'flags': {
                'enable_deformation': self.enable_deformation,
                'enable_inpainting': self.enable_inpainting
            }
        })

    async def _handle_set_debug_mode(self, data: dict):
        """Enable or disable debug mode"""
        enabled = bool(data.get('enabled', False))
        self.renderer.set_debug_mode(enabled)

        print(f"[Session {self.session_id}] ✓ Debug mode: {'ON' if enabled else 'OFF'}")

        await self.send_message({
            'type': 'debug_mode_updated',
            'enabled': enabled
        })

        # Re-render with debug mode
        await self.send_render()

    async def _handle_set_debug_options(self, data: dict):
        """Update debug visualization options"""
        options = data.get('options', {})

        # Update renderer debug options
        self.renderer.set_debug_options(options)

        print(f"[Session {self.session_id}] ✓ Debug options updated")

        await self.send_message({
            'type': 'debug_options_updated',
            'options': options
        })

        # Re-render with new debug options
        await self.send_render()

    async def _handle_convert_brush_from_image(self, data: dict):
        """Convert uploaded image to 3DGS brush"""
        import asyncio
        from backend.api.upload import get_upload_image, update_upload_status
        from backend.core.brush_converter import BrushConverter

        upload_id = data.get('upload_id')
        image_data = data.get('image_data')
        depth_profile = data.get('depth_profile', 'flat')  # Default to flat for 2D brush images
        depth_scale = data.get('depth_scale', 0.2)

        # Check if we have either upload_id or direct image_data
        if not upload_id and not image_data:
            await self.send_error("No upload_id or image_data provided")
            return

        if upload_id:
            print(f"[Session {self.session_id}] Converting brush from upload {upload_id}")
        else:
            print(f"[Session {self.session_id}] Converting brush from direct image data")

        try:
            # Send initial progress
            await self.send_message({
                'type': 'conversion_progress',
                'upload_id': upload_id,
                'progress': 10,
                'status': 'Loading image...'
            })

            # Get image either from direct data or upload storage
            image = None

            # Check if image data was provided directly (for testing/direct conversion)
            image_data = data.get('image_data')
            if image_data and image_data.startswith('data:image'):
                # Extract base64 data from data URI
                import base64
                from PIL import Image
                import io

                # Remove data URI prefix
                base64_str = image_data.split(',')[1]
                image_bytes = base64.b64decode(base64_str)

                # Load image
                pil_image = Image.open(io.BytesIO(image_bytes))
                if pil_image.mode != 'RGBA':
                    pil_image = pil_image.convert('RGBA')

                # Convert to numpy array
                image = np.array(pil_image)
                print(f"[Session {self.session_id}] Using direct image data: {image.shape}")
            else:
                # Get uploaded image from storage
                image = get_upload_image(upload_id)
                if image is None:
                    await self.send_error(f"Upload {upload_id} not found")
                    return

            # Update progress
            await self.send_message({
                'type': 'conversion_progress',
                'upload_id': upload_id,
                'progress': 20,
                'status': 'Estimating depth...'
            })

            # Create converter (run in thread pool to avoid blocking)
            converter = BrushConverter(
                use_midas=False,  # Use heuristic for now (faster)
                target_gaussian_count=800  # Increased for better texture detail with aspect ratio fix
            )

            # Update progress
            await self.send_message({
                'type': 'conversion_progress',
                'upload_id': upload_id,
                'progress': 40,
                'status': 'Generating point cloud...'
            })

            # Convert to 3DGS brush (run in thread pool)
            loop = asyncio.get_event_loop()
            brush_name = f"brush_{upload_id[:8]}" if upload_id else "direct_brush"
            brush_stamp = await loop.run_in_executor(
                None,
                converter.convert_2d_to_3dgs,
                image,
                brush_name,
                depth_profile,
                depth_scale,
                0  # No optimization for now
            )

            # Update progress
            await self.send_message({
                'type': 'conversion_progress',
                'upload_id': upload_id,
                'progress': 80,
                'status': 'Creating brush...'
            })

            # Store as current brush
            self.brush = brush_stamp
            gaussian_count = len(brush_stamp.gaussians)

            # Recreate painter with new brush
            self.current_painter = StrokePainter(self.brush, self.scene_gaussians)

            print(f"[Session {self.session_id}] ✓ Created brush with {gaussian_count} Gaussians")

            # Auto-save converted brush to library
            from backend.core.brush_manager import get_brush_manager
            brush_manager = get_brush_manager()
            brush_id = brush_manager.save_brush(
                brush_stamp,
                name=brush_stamp.metadata.get('name', 'Converted Brush'),
                brush_type='converted',
                source='image'
            )

            # Update upload status (only if we have an upload_id)
            if upload_id:
                update_upload_status(upload_id, 'completed', 100)

            # Prepare brush data for frontend preview
            brush_data = {
                'gaussians': [
                    {
                        'position': g.position.tolist() if hasattr(g.position, 'tolist') else g.position,
                        'scale': g.scale.tolist() if hasattr(g.scale, 'tolist') else g.scale,
                        'rotation': g.rotation.tolist() if hasattr(g.rotation, 'tolist') else g.rotation,
                        'opacity': float(g.opacity),
                        'color': g.color.tolist() if hasattr(g.color, 'tolist') else g.color
                    }
                    for g in brush_stamp.gaussians  # Send all Gaussians for full detail
                ],
                'metadata': brush_stamp.metadata,
                'center': brush_stamp.center.tolist() if hasattr(brush_stamp.center, 'tolist') else brush_stamp.center,
                'size': float(brush_stamp.size),
                'spacing': float(brush_stamp.spacing)
            }

            # Send completion message
            await self.send_message({
                'type': 'conversion_complete',
                'upload_id': upload_id,
                'brush_name': brush_stamp.metadata.get('name', 'converted_brush'),
                'gaussian_count': gaussian_count,
                'progress': 100,
                'brush_data': brush_data
            })

            # Send updated stats
            await self.send_stats()

        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            print(f"[Session {self.session_id}] ✗ {error_msg}")

            # Update upload status (only if we have an upload_id)
            if upload_id:
                update_upload_status(upload_id, 'failed', 0)

            await self.send_message({
                'type': 'conversion_failed',
                'upload_id': upload_id,
                'error': error_msg
            })

    async def send_render(self):
        """Render current scene and send to client"""
        try:
            # Render
            image = self.renderer.render(self.scene_gaussians)

            # Convert to base64 JPEG (4-6x faster than PNG)
            img_base64 = numpy_to_base64_jpeg(image, quality=85)

            # Send to client
            await self.send_message({
                'type': 'render_update',
                'image': img_base64,
                'width': self.renderer.width,
                'height': self.renderer.height
            })

        except Exception as e:
            print(f"[Session {self.session_id}] ✗ Render error: {e}")
            traceback.print_exc()

    async def send_stats(self):
        """Send scene statistics to client"""
        await self.send_message({
            'type': 'stats',
            'num_gaussians': len(self.scene_gaussians),
            'num_strokes': len(self.current_painter.placed_stamps)
        })

    async def send_message(self, data: dict):
        """Send JSON message to client"""
        try:
            await self.ws.send_json(data)
        except Exception as e:
            print(f"Error sending message: {e}")

    async def send_error(self, error_msg: str):
        """Send error message to client"""
        await self.send_message({
            'type': 'error',
            'message': error_msg
        })

    # Brush Library Management Handlers
    async def _handle_save_brush(self, data: dict):
        """Save current brush to library"""
        from backend.core.brush_manager import get_brush_manager

        if self.brush is None:
            await self.send_error("No active brush to save")
            return

        brush_manager = get_brush_manager()
        name = data.get('name', 'Unnamed Brush')
        brush_type = data.get('type', 'custom')
        source = data.get('source', 'manual')

        try:
            # Save the brush
            brush_id = brush_manager.save_brush(
                self.brush,
                name=name,
                brush_type=brush_type,
                source=source
            )

            await self.send_message({
                'type': 'brush_saved',
                'brush_id': brush_id,
                'name': name
            })

            # Send updated brush list
            await self._handle_list_brushes()

        except Exception as e:
            await self.send_error(f"Failed to save brush: {str(e)}")

    async def _handle_load_brush(self, data: dict):
        """Load brush from library"""
        from backend.core.brush_manager import get_brush_manager

        brush_id = data.get('brush_id')
        if not brush_id:
            await self.send_error("No brush_id provided")
            return

        brush_manager = get_brush_manager()

        try:
            # Load the brush
            brush = brush_manager.load_brush(brush_id)
            if brush is None:
                await self.send_error(f"Brush {brush_id} not found")
                return

            # Set as current brush
            self.brush = brush

            # Recreate painter with new brush
            self.current_painter = StrokePainter(self.brush, self.scene_gaussians)

            # Get brush info
            brush_info = brush_manager.get_brush_info(brush_id)

            await self.send_message({
                'type': 'brush_loaded',
                'brush_id': brush_id,
                'brush_info': brush_info
            })

            # Send a render update
            await self.send_render()

        except Exception as e:
            await self.send_error(f"Failed to load brush: {str(e)}")

    async def _handle_list_brushes(self):
        """List all available brushes"""
        from backend.core.brush_manager import get_brush_manager

        brush_manager = get_brush_manager()

        try:
            brushes = brush_manager.list_brushes()

            await self.send_message({
                'type': 'brush_list',
                'brushes': brushes
            })

        except Exception as e:
            await self.send_error(f"Failed to list brushes: {str(e)}")

    async def _handle_delete_brush(self, data: dict):
        """Delete brush from library"""
        from backend.core.brush_manager import get_brush_manager

        brush_id = data.get('brush_id')
        if not brush_id:
            await self.send_error("No brush_id provided")
            return

        brush_manager = get_brush_manager()

        try:
            # Delete the brush
            success = brush_manager.delete_brush(brush_id)

            if success:
                await self.send_message({
                    'type': 'brush_deleted',
                    'brush_id': brush_id
                })

                # Send updated brush list
                await self._handle_list_brushes()
            else:
                await self.send_error(f"Brush {brush_id} not found")

        except Exception as e:
            await self.send_error(f"Failed to delete brush: {str(e)}")

    async def _handle_update_brush_metadata(self, data: dict):
        """Update brush metadata (name, tags, etc.)"""
        from backend.core.brush_manager import get_brush_manager

        brush_id = data.get('brush_id')
        if not brush_id:
            await self.send_error("No brush_id provided")
            return

        brush_manager = get_brush_manager()

        try:
            # Update metadata
            updates = {}
            if 'name' in data:
                updates['name'] = data['name']
            if 'tags' in data:
                updates['tags'] = data['tags']

            success = brush_manager.update_brush_metadata(brush_id, **updates)

            if success:
                await self.send_message({
                    'type': 'brush_metadata_updated',
                    'brush_id': brush_id,
                    'updates': updates
                })

                # Send updated brush list
                await self._handle_list_brushes()
            else:
                await self.send_error(f"Brush {brush_id} not found")

        except Exception as e:
            await self.send_error(f"Failed to update brush metadata: {str(e)}")


class ConnectionManager:
    """
    Manage WebSocket connections

    여러 클라이언트 연결 관리
    """

    def __init__(self):
        self.active_sessions: Dict[str, PaintingSession] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> PaintingSession:
        """
        Accept new WebSocket connection

        Args:
            websocket: WebSocket connection
            session_id: Unique session ID

        Returns:
            PaintingSession instance
        """
        print(f"[ConnectionManager] ✓ Accepting WebSocket connection for session {session_id}")
        await websocket.accept()

        print(f"[ConnectionManager] Creating painting session")
        session = PaintingSession(websocket, session_id)
        self.active_sessions[session_id] = session

        print(f"[ConnectionManager] ✓ Session {session_id} connected. Total sessions: {len(self.active_sessions)}")

        return session

    def disconnect(self, session_id: str):
        """
        Remove session

        Args:
            session_id: Session ID to remove
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            print(f"Session {session_id} disconnected. Total sessions: {len(self.active_sessions)}")

    async def handle_session(self, session: PaintingSession):
        """
        Handle WebSocket session

        Args:
            session: PaintingSession instance
        """
        try:
            while True:
                # Receive message
                data = await session.ws.receive_json()

                # Handle message
                await session.handle_message(data)

        except WebSocketDisconnect:
            print(f"Session {session.session_id} disconnected")
            self.disconnect(session.session_id)
        except Exception as e:
            print(f"Session {session.session_id} error: {e}")
            traceback.print_exc()
            self.disconnect(session.session_id)


# Global connection manager
manager = ConnectionManager()
