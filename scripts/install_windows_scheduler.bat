@echo off
REM Registers Windows Scheduled Task to run every morning at 07:00 AM
echo ================================================================================
echo   INSTALLING AUTOMATED DAILY BANGALORE VEGETABLE PRICE FORECAST TASK
echo ================================================================================
echo.

set "SCRIPT_DIR=%~dp0"
set "BAT_PATH=%SCRIPT_DIR%run_daily.bat"

echo Task Name : BangaloreVegetablePriceForecast
echo Action    : %BAT_PATH%
echo Schedule  : Daily at 07:00 AM
echo.

schtasks /create /tn "BangaloreVegetablePriceForecast" /tr "\"%BAT_PATH%\"" /sc daily /st 07:00 /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Daily automated forecast scheduled successfully!
    echo It will run every day at 07:00 AM automatically.
) else (
    echo.
    echo [NOTE] If you saw an access denied message, please right-click this .bat file and select 'Run as administrator'.
)

echo.
pause
