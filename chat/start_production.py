import sys
import os

# Change to chat directory
os.chdir('/var/www/slimpunt.nl/v5JL/chat')

# Add chat directory to path
sys.path.insert(0, '/var/www/slimpunt.nl/v5JL/chat')

from flask import Flask, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Get the chat directory
CHAT_DIR = '/var/www/slimpunt.nl/v5JL/chat'

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
    print("v5JL Chat UI - Production")
    print("=" * 50)
    print("Frontend running on: http://localhost:5011")
    print("=" * 50)

    app.run(
        host='0.0.0.0',
        port=5011,
        debug=False,
        use_reloader=False
    )
