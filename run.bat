@echo off
echo AFK Tracker Launcher
echo ==================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in the PATH
    echo Please install Python 3.7 or higher
    pause
    exit /b 1
)

REM Check if this is the first run
if not exist models\haarcascade_frontalface_default.xml (
    echo First run detected. Downloading required models...
    python download_models.py
    if %errorlevel% neq 0 (
        echo Error downloading models
        pause
        exit /b 1
    )
)

REM Show menu
:menu
echo.
echo Choose an option:
echo 1. Run Basic AFK Detector
echo 2. Run Debug Menu (with visualization options)
echo 3. Download/Update Models
echo 4. Exit
echo.

set /p option=Enter option number: 

if "%option%"=="1" (
    echo Starting Basic AFK Detector...
    python run.py
    goto menu
) else if "%option%"=="2" (
    echo Starting Debug Menu...
    python run.py --debug
    goto menu
) else if "%option%"=="3" (
    echo Downloading/Updating models...
    python download_models.py
    goto menu
) else if "%option%"=="4" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid option
    goto menu
) 