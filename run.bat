@echo off
:: Check if we are in the right folder, otherwise cd into it
if not exist main.py (
    if exist VisualErgonomicsAssessment-main (
        cd VisualErgonomicsAssessment-main
    )
)

echo Starting Visual Ergonomics System...

:: Activate venv if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    python main.py
) else (
    echo [ERROR] Virtual environment not found. Please run setup.bat first.
    pause
)
