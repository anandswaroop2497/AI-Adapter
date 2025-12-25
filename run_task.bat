@echo off
REM Helper script to run the Kavach runner with the virtual environment

echo ========================================
echo Kavach Camera Runner
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

REM Check if adapter is running
echo Checking if adapter is running...
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:9100/health' -UseBasicParsing -TimeoutSec 2 | Out-Null; Write-Host 'Adapter is running!' -ForegroundColor Green } catch { Write-Host 'WARNING: Adapter not responding on port 9100' -ForegroundColor Yellow; Write-Host 'Start it with: uvicorn adapter.main:app --reload --port 9100' -ForegroundColor Yellow }"
echo.

REM Run the runner with arguments
venv\Scripts\python.exe kavach\runner.py %*
