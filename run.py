#!/usr/bin/env python
import argparse
import os
from afk_detector import AFKDetector
from debug_menu import DebugMenuAFKDetector

def main():
    parser = argparse.ArgumentParser(description='AFK Tracker - Detect when you are away from keyboard')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    parser.add_argument('--threshold', type=float, default=3.0, 
                        help='Time in seconds without face detection to consider AFK (default: 3.0)')
    parser.add_argument('--history', type=int, default=30,
                        help='Number of frames to use for detection smoothing (default: 30)')
    parser.add_argument('--debug', action='store_true', help='Launch with debug menu interface')
    
    args = parser.parse_args()
    
    if args.debug:
        # Launch with debug menu
        detector = DebugMenuAFKDetector(
            camera_index=args.camera,
            afk_threshold=args.threshold,
            history_size=args.history
        )
        
        print("AFK Tracker Debug Menu starting...")
        print("Controls:")
        print("  1: Toggle Raw Camera")
        print("  2: Toggle Grayscale")
        print("  3: Toggle Thresholded View")
        print("  4: Toggle Face Detection")
        print("  5: Toggle Eye Detection")
        print("  6: Toggle Stats Display")
        print("  7: Reset History")
        print("  q: Quit")
    else:
        # Launch normal detector
        detector = AFKDetector(
            camera_index=args.camera,
            afk_threshold=args.threshold,
            history_size=args.history
        )
        
        print("AFK Tracker starting...")
        print("Press 'q' to quit")
    
    detector.run()

if __name__ == "__main__":
    main() 