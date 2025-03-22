#!/bin/bash

echo "AFK Tracker Launcher"
echo "=================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in the PATH"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

# Check if this is the first run
if [ ! -f "models/haarcascade_frontalface_default.xml" ]; then
    echo "First run detected. Downloading required models..."
    python3 download_models.py
    if [ $? -ne 0 ]; then
        echo "Error downloading models"
        exit 1
    fi
fi

# Show menu
while true; do
    echo
    echo "Choose an option:"
    echo "1. Run Basic AFK Detector"
    echo "2. Run Debug Menu (with visualization options)"
    echo "3. Download/Update Models"
    echo "4. Exit"
    echo
    
    read -p "Enter option number: " option
    
    case $option in
        1)
            echo "Starting Basic AFK Detector..."
            python3 run.py
            ;;
        2)
            echo "Starting Debug Menu..."
            python3 run.py --debug
            ;;
        3)
            echo "Downloading/Updating models..."
            python3 download_models.py
            ;;
        4)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option"
            ;;
    esac
done 