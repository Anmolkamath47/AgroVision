@echo off
REM Bangalore Vegetable Price Forecasting - Daily Automated Run
cd /d "%~dp0\.."
echo [INFO] Running Daily Bangalore Grocery Price Forecast & Verification Pipeline...
python src/continuous_pipeline.py --once
echo.
echo [INFO] Daily Cycle Completed. Press any key to close.
pause
