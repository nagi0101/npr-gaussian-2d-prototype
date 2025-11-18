"""
FastAPI main application

3DGS 2D Painting Backend Server
"""

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
import os

from backend.api.websocket import manager
from backend.api.upload import router as upload_router
from backend.config import config

# Create FastAPI app
app = FastAPI(
    title="3DGS 2D Painting Server",
    description="Backend server for Gaussian Splatting brush painting",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include upload router
app.include_router(upload_router)

# Mount static files (frontend)
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main page"""
    index_path = os.path.join(frontend_dir, "index.html")

    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return """
        <html>
            <head><title>3DGS 2D Painting</title></head>
            <body>
                <h1>3DGS 2D Painting Server</h1>
                <p>Frontend not found. Please check frontend directory.</p>
                <p>WebSocket endpoint: ws://localhost:8000/ws</p>
            </body>
        </html>
        """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "3DGS 2D Painting",
        "version": "0.1.0",
        "active_sessions": len(manager.active_sessions)
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for painting

    클라이언트는 이 엔드포인트로 연결하여 실시간 페인팅 진행
    """
    # Generate unique session ID
    session_id = str(uuid.uuid4())

    # Connect
    session = await manager.connect(websocket, session_id)

    # Initialize brush library with default brushes
    from backend.core.brush_manager import get_brush_manager
    brush_manager = get_brush_manager()
    brush_manager.create_default_brushes()

    # Get initial brush list
    brushes = brush_manager.list_brushes()

    # Send initial message with brush list
    await session.send_message({
        'type': 'connected',
        'session_id': session_id,
        'config': {
            'width': config.RENDER_WIDTH,
            'height': config.RENDER_HEIGHT,
            'world_min': config.WORLD_MIN,
            'world_max': config.WORLD_MAX
        },
        'brushes': brushes
    })

    # Send initial render (empty scene)
    await session.send_render()

    # Handle session
    await manager.handle_session(session)


def main():
    """Run the server"""
    print(f"""
    ╔═══════════════════════════════════════════════╗
    ║   3DGS 2D Painting Server                     ║
    ║                                               ║
    ║   Server: http://{config.HOST}:{config.PORT}            ║
    ║   WebSocket: ws://{config.HOST}:{config.PORT}/ws      ║
    ║                                               ║
    ║   Press Ctrl+C to stop                        ║
    ╚═══════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "backend.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )


if __name__ == "__main__":
    main()
