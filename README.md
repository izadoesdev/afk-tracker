# AFK Tracker

A Python application that uses your webcam to detect when you're away from your keyboard (AFK) by tracking face presence and eye visibility.

## Features

- Face detection to determine presence at the computer
- Eye detection to determine if you're looking away 
- Blink detection to avoid false AFK triggers during normal blinking
- Glasses detection for improved eye tracking with eyewear
- AFK status tracking with timestamps
- Statistical summaries of AFK sessions
- Real-time visualization with status indicators

### Debug Menu Features

The debug menu adds:
- Visualization of the face detection process
- Ability to toggle between different view modes (raw camera, grayscale, thresholded)
- Face detection heatmap showing historical detections
- Real-time adjustment of detection parameters
- Detailed eye detection visualization
- Blink detection visualization with history
- Glasses detection status and confidence
- Face detection confidence metrics
- Blink counter with estimated blink rate 

## Requirements

- Python 3.7+
- Webcam
- OpenCV with Haar Cascades

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/afk-tracker.git
   cd afk-tracker
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application launcher:
   - Windows: `run.bat`
   - Linux/Mac: `./run.sh`

## Usage

### Basic Detector

Run the basic application:

```
python run.py
```

Options:
- `--camera` - Camera index to use (default: 0)
- `--threshold` - Time in seconds without face to consider AFK (default: 3.0)
- `--history` - Frames to use for detection smoothing (default: 30)

### Debug Menu

Run the application with debug menu:

```
python run.py --debug
```

Debug menu options:
- `1` - Toggle Raw Camera View
- `2` - Toggle Grayscale View
- `3` - Toggle Thresholded View
- `4` - Toggle Face Detection Visualization
- `5` - Toggle Eye Detection Visualization
- `6` - Toggle Blink Detection Visualization 
- `7` - Toggle Glasses Detection Visualization
- `8` - Toggle Stats Display
- `9` - Reset History
- `0` - Reset Blink Counter
- Use the threshold sliders to adjust detection sensitivity

### Controls
- Press 'q' to quit the application

### Display Information
- Green "Status: Present" - You are detected at your computer
- Red "Status: AFK" - You are detected as away from keyboard
- Blue rectangle - Face detected
- Red rectangle - Face detected but looking away
- Green rectangle - Eyes detected within face region
- Purple "Blinking" label - Detected blink in progress
- Yellow "Glasses: Detected/Not Detected" - Current glasses detection status
- Red "No face" timer - How long since a face was last detected
- Total AFK time - Cumulative time spent away from keyboard

## How it Works

### Detection Process
The application uses Haar Cascade classifiers to detect faces and eyes:
1. Captures video from your webcam
2. Converts frames to grayscale for processing
3. Applies face detection algorithms to find faces
4. For each detected face, searches for eyes (using both regular and glasses-specific eye detection)
5. Differentiates between blinking and looking away based on detection patterns
6. Considers user AFK if no face is detected for the threshold time
7. Tracks statistics about AFK sessions
8. Provides visual feedback about detection status

### Intelligent Blink Detection
The application can distinguish between normal blinking and looking away:
1. Tracks the duration of eye disappearance
2. Short durations (< 0.5 seconds) are classified as blinks
3. Longer durations are classified as looking away
4. A history buffer smooths detection to prevent false triggers
5. Looking away must persist for 8 seconds (configurable) before triggering AFK status

### Glasses Detection
The system automatically detects if you're wearing glasses:
1. Uses specialized Haar cascade for detecting eyes with glasses
2. Dynamically switches between regular and glasses-specific eye detection
3. Maintains a confidence score for glasses detection
4. Automatically adapts detection parameters based on eyewear

### Debug Visualization
The debug menu provides insights into how the detection works:
1. Face detection history displayed as a heatmap overlay
2. Different view modes to see each processing step
3. Real-time statistics about detection confidence
4. Visual elements showing exactly which features are being detected
5. Blink history visualization showing recent blink pattern
6. Confidence metrics for all detection aspects (face, eyes, blinks, glasses)

## Customization

You can modify the detector parameters to adjust sensitivity and behavior:
- Adjust the threshold sliders in the debug menu:
  - Blink Duration - How long an eye disappearance is considered a blink vs. looking away
  - Looking Away Threshold - How long you must look away before triggering AFK
  - Face/Eye Scale Factors - Detection sensitivity for faces and eyes
  - Min Neighbors - Filtering for detection quality (higher = fewer false positives)
- Change command line parameters for AFK threshold and history size

## License

MIT 