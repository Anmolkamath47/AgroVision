#!/usr/bin/env bash
set -e

echo "========================================================================="
echo "   AGROVISION ANALYTICS: PRODUCTION DEPLOYMENT RUNNER"
echo "========================================================================="

# Detect Python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "[1/3] Installing / verifying production dependencies..."
$PYTHON_CMD -m pip install -q --upgrade pip
$PYTHON_CMD -m pip install -q -r requirements.txt
$PYTHON_CMD -m pip install -q gunicorn

echo "[2/3] Checking database connectivity..."
$PYTHON_CMD -c "from src.database import init_db; init_db()" || echo "Notice: Database will operate in resilient CSV fallback mode if PostgreSQL is unavailable."

echo "[3/3] Launching production WSGI server..."
exec $PYTHON_CMD scripts/run_production.py
