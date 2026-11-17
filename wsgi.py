"""
WSGI Production Entry Point for AgroVision Intelligence Platform.
Compatible with Gunicorn, Waitress, uWSGI, and cloud container runtimes.
"""

import os
import sys

# Ensure project root is in Python module path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app

# Expose application object for WSGI runners
application = app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Try waitress first for production serving
    try:
        from waitress import serve
        print(f"[AgroVision WSGI] Serving with Waitress on http://{host}:{port}")
        serve(app, host=host, port=port, threads=4)
    except ImportError:
        print(f"[AgroVision WSGI] Serving with Flask development server on http://{host}:{port}")
        app.run(host=host, port=port, debug=False)
