@echo off
echo ========================================
echo   Visual Ergonomics System - SETUP
echo ========================================

:: Check if venv exists, if not create it
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
echo Installing/Updating dependencies...
pip install -r requirements.txt

echo.
echo SETUP COMPLETE!
echo You can now use 'run.bat' for fast startup.
echo.
pause
