#!/bin/bash

# AI Jupyter Notebook v5 - JupyterLab Edition Startup Script

cd "$(dirname "$0")"

echo "===================================================="
echo "  Starting AI Jupyter Notebook v5 - JupyterLab"
echo "===================================================="
echo ""

# Kill any existing processes on our ports
echo "Cleaning up existing processes..."
lsof -ti:5000 | xargs kill -9 2>/dev/null
lsof -ti:5001 | xargs kill -9 2>/dev/null
lsof -ti:8888 | xargs kill -9 2>/dev/null

echo ""
echo "Starting services..."
echo ""

# Start Flask backend in background
echo "1/3 Starting Flask API backend (port 5000)..."
cd backend
poetry run python app.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 2

# Start Chat UI in background
echo "2/3 Starting AI Chat UI (port 5001)..."
cd chat
poetry run python server.py &
CHAT_PID=$!
cd ..

sleep 1

# Start JupyterLab
echo "3/3 Starting JupyterLab (port 8888)..."
echo ""
echo "===================================================="
echo "  All services started!"
echo "===================================================="
echo "  JupyterLab:  http://localhost:8888/lab"
echo "  AI Chat:     http://localhost:5001"
echo "  Backend API: http://localhost:5000"
echo "===================================================="
echo ""
echo "TIP: Open JupyterLab and AI Chat in split-screen!"
echo ""
poetry run jupyter lab --config=jupyter_lab_config.py

# When JupyterLab exits, kill all services
echo ""
echo "Shutting down all services..."
kill $BACKEND_PID 2>/dev/null
kill $CHAT_PID 2>/dev/null
lsof -ti:5000 | xargs kill -9 2>/dev/null
lsof -ti:5001 | xargs kill -9 2>/dev/null
lsof -ti:8888 | xargs kill -9 2>/dev/null
echo "All services stopped."
