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
            prefer_gpu=True  # Try GPU first, fallback to CPU
        )
        self.renderer.set_world_bounds(
            np.array(config.WORLD_MIN),
            np.array(config.WORLD_MAX)
        )

        # Scene state
        self.scene_gaussians: list[Gaussian2D] = []
        self.current_painter: Optional[StrokePainter] = None
        self.brush: Optional[BrushStamp] = None

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
            elif msg_type == 'request_render':
                await self.send_render()
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

        # Start stroke
        self.current_painter.start_stroke(position_3d, normal)

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
        self.current_painter.finish_stroke()

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
        """Set brush parameters"""
        if 'spacing' in data:
            self.brush.spacing = float(data['spacing'])

        if 'size' in data:
            size_factor = float(data['size'])
            # Scale all Gaussians in brush
            for g in self.brush.gaussians:
                g.scale *= size_factor

        if 'opacity' in data:
            opacity = float(data['opacity'])
            self.brush.set_opacity(opacity)

        if 'color' in data:
            color = np.array(data['color'], dtype=np.float32)
            self.brush.set_color(color)

        await self.send_message({
            'type': 'brush_params_updated',
            'params': {
                'spacing': self.brush.spacing,
                'size': self.brush.size,
                'opacity': self.brush.gaussians[0].opacity if len(self.brush.gaussians) > 0 else 0.8
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

        # Recreate painter with new brush
        self.current_painter = StrokePainter(self.brush, self.scene_gaussians)

        await self.send_message({
            'type': 'brush_created',
            'pattern': pattern,
            'num_gaussians': len(self.brush.gaussians)
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
