"""
AgroVision Production Server Launcher.
Automatically selects Waitress (Windows) or Gunicorn (Linux/macOS).
Ensures database connectivity, model availability, and logs telemetry.
"""

import os
import sys
import platform
import logging

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgroVisionProduction")


def run():
    from src.database import check_db_connection, init_db
    from app import app

    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "0.0.0.0" if platform.system() != "Windows" else "127.0.0.1")

    print("\n" + "=" * 76)
    print("      AGROVISION ANALYTICS: PRODUCTION WSGI SERVER STARTUP")
    print(f"      Platform: {platform.system()} {platform.release()} | Python: {platform.python_version()}")
    print("=" * 76)

    # 1. Initialize database if connected
    db_status = check_db_connection()
    if db_status.get("connected"):
        logger.info(f"Connected to PostgreSQL ({db_status['host']}:{db_status['port']}/{db_status['database']}). Initializing schema...")
        init_db()
    else:
        logger.warning(f"PostgreSQL not connected ({db_status.get('error', 'Offline')}). Operating with resilient CSV fallback store.")

    # 2. Check models directory
    models_dir = os.path.join(BASE_DIR, "models")
    if os.path.exists(models_dir):
        model_count = len([f for f in os.listdir(models_dir) if f.endswith(".json")])
        logger.info(f"Verified {model_count} trained machine learning models in models/")
    else:
        logger.warning("Models directory not found.")

    # 3. Serve with appropriate WSGI server
    if platform.system() == "Windows":
        try:
            from waitress import serve
            logger.info(f"Starting Waitress multi-threaded production WSGI server on http://{host}:{port}")
            print(f"\n>>> Application Live & Ready: http://{host}:{port}\n")
            serve(app, host=host, port=port, threads=6, connection_limit=200, channel_timeout=60)
        except ImportError:
            logger.warning("Waitress not found. Falling back to Flask server.")
            app.run(host=host, port=port, debug=False)
    else:
        try:
            from gunicorn.app.base import BaseApplication

            class StandaloneApplication(BaseApplication):
                def __init__(self, application, options=None):
                    self.options = options or {}
                    self.application = application
                    super().__init__()

                def load_config(self):
                    for key, value in self.options.items():
                        if key in self.cfg.settings and value is not None:
                            self.cfg.set(key.lower(), value)

                def load(self):
                    return self.application

            options = {
                "bind": f"{host}:{port}",
                "workers": 2,
                "threads": 4,
                "timeout": 120,
                "accesslog": "-",
                "errorlog": "-",
            }
            logger.info(f"Starting Gunicorn multi-process production server on http://{host}:{port}")
            print(f"\n>>> Application Live & Ready: http://{host}:{port}\n")
            StandaloneApplication(app, options).run()
        except ImportError:
            logger.warning("Gunicorn not found. Falling back to Waitress / Flask.")
            try:
                from waitress import serve
                serve(app, host=host, port=port, threads=4)
            except ImportError:
                app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run()
