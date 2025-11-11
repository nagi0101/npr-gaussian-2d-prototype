#!/bin/bash
# Background deployment script for 3DGS 2D Painting Backend
# Runs server in background with nohup

set -e

echo "======================================"
echo "  Background Deployment"
echo "======================================"

ENV_NAME="gaussian-brush-2d"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Check if server is already running
if pgrep -f "backend.main" > /dev/null; then
    echo "⚠️  Server appears to be already running"
    read -p "Kill existing server and restart? (y/N): " kill_existing

    if [[ $kill_existing =~ ^[Yy]$ ]]; then
        echo "Stopping existing server..."
        pkill -f "backend.main"
        sleep 2
    else
        echo "Aborted."
        exit 1
    fi
fi

# Start server in background
echo "Starting server in background..."
nohup python -m backend.main > server.log 2>&1 &

SERVER_PID=$!
echo "Server started with PID: ${SERVER_PID}"
echo "Log file: server.log"

# Wait a moment and check if server is running
sleep 2

if ps -p ${SERVER_PID} > /dev/null; then
    echo "✓ Server is running"
    echo ""
    echo "Commands:"
    echo "  View logs:    tail -f server.log"
    echo "  Stop server:  kill ${SERVER_PID}"
    echo "  Or:           pkill -f backend.main"
    echo ""
    echo "Server URL: http://0.0.0.0:8000"
else
    echo "❌ Server failed to start"
    echo "Check server.log for errors"
    cat server.log
    exit 1
fi
