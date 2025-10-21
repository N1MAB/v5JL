"""
Simple web server for AI Chat Interface
Serves the chat HTML on port 5001
"""

from flask import Flask, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Get the directory where this script is located
CHAT_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    """Serve the v4-style interface"""
    return send_from_directory(CHAT_DIR, 'v4style.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (JS, CSS, etc.)"""
    return send_from_directory(CHAT_DIR, filename)

if __name__ == '__main__':
    print("=" * 50)
    print("AI Notebook v5 - Enhanced Workflow")
    print("=" * 50)
    print("UI: http://localhost:5001")
    print("Backend API: http://localhost:5000")
    print("\nOpen http://localhost:5001 in your browser")
    print("v4 workflow + manual execution control!")
    print("=" * 50)

    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False
    )
