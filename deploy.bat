@echo off
title AgroVision Analytics - Production Deploy Launcher
echo =========================================================================
echo    AGROVISION ANALYTICS: ONE-CLICK PRODUCTION DEPLOYMENT
echo =========================================================================
echo.

:: Check Python installation
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set PYTHON_CMD=python
    ) else (
        echo [ERROR] Python not found in system PATH. Please install Python 3.10+.
        pause
        exit /b 1
    )
)

echo [1/3] Checking and installing production dependencies...
%PYTHON_CMD% -m pip install -q -r requirements.txt
%PYTHON_CMD% -m pip install -q waitress

echo [2/3] Checking database connectivity and schema...
%PYTHON_CMD% -c "from src.database import init_db; init_db()"

echo [3/3] Launching AgroVision WSGI production server on http://127.0.0.1:5001 ...
echo.
%PYTHON_CMD% scripts/run_production.py
pause
