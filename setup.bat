@echo off
REM Simple-Comfort setup — run once before first launch (Windows).
REM Creates a Python virtual environment and installs all dependencies.
cd /d "%~dp0"
python -m venv venv
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete. Run launch.bat to start Simple-Comfort.
pause
