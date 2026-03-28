@echo off
chcp 65001 >nul
echo ==========================================
echo Industrial Defect Detection System
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check dependencies
echo [1/3] Checking dependencies...
pip show pyyaml >nul 2>&1
if errorlevel 1 (
    echo Installing PyYAML...
    pip install pyyaml -q
)

REM Create directories
echo [2/3] Creating directory structure...
if not exist input_images mkdir input_images
if not exist templates mkdir templates
if not exist output mkdir output
if not exist logs mkdir logs
if not exist sounds mkdir sounds

REM Check config file
if not exist config.yaml (
    echo [WARNING] Config file not found
)

REM Run detection system
echo [3/3] Starting detection system...
echo.
echo Usage:
echo   - Put images to detect in input_images folder
echo   - Put template image as template.jpg in templates folder
echo   - Press Q to quit
echo   - Press SPACE to pause
echo.

REM Run from current directory
python detector.py --config config.yaml

if errorlevel 1 (
    echo.
    echo [ERROR] Program failed
    pause
) else (
    echo.
    echo [DONE] Detection complete, results saved in output folder
    pause
)
