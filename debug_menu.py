#!/usr/bin/env python
import cv2
import numpy as np
import time
import datetime
import argparse
import os
from collections import deque
from afk_detector import AFKDetector

class DebugMenuAFKDetector(AFKDetector):
    def __init__(self, 
                 camera_index=0, 
                 face_cascade_path='haarcascade_frontalface_default.xml',
                 eye_cascade_path='haarcascade_eye.xml',
                 afk_threshold=3.0,
                 history_size=30):
        """Extended AFKDetector with debug visualization options."""
        super().__init__(camera_index, face_cascade_path, eye_cascade_path, afk_threshold, history_size)
        
        # Debug options
        self.show_raw_camera = True
        self.show_grayscale = False
        self.show_thresholded = False
        self.show_face_detection = True
        self.show_eye_detection = True
        self.show_stats = True
        self.show_blink_detection = True
        self.show_glasses_detection = True
        self.detection_threshold = 90  # for thresholding grayscale
        
        # Setup debug window
        cv2.namedWindow('AFK Detector Debug', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('AFK Detector Debug', 800, 600)
        
        # Create trackbars for adjusting detection parameters
        cv2.createTrackbar('Threshold', 'AFK Detector Debug', self.detection_threshold, 255, self._on_threshold_change)
        cv2.createTrackbar('Face Scale ×10', 'AFK Detector Debug', int(self.face_scale_factor*10), 20, self._on_face_scale_change)
        cv2.createTrackbar('Face MinNeigh', 'AFK Detector Debug', self.face_min_neighbors, 10, self._on_face_minneigh_change)
        cv2.createTrackbar('Eye Scale ×10', 'AFK Detector Debug', int(self.eye_scale_factor*10), 20, self._on_eye_scale_change)
        cv2.createTrackbar('Eye MinNeigh', 'AFK Detector Debug', self.eye_min_neighbors, 10, self._on_eye_minneigh_change)
        cv2.createTrackbar('Blink Duration x10', 'AFK Detector Debug', int(self.blink_duration_threshold*10), 30, self._on_blink_duration_change)
        cv2.createTrackbar('Looking Away Threshold', 'AFK Detector Debug', int(self.looking_away_threshold), 20, self._on_looking_away_threshold_change)
        
        # History of face detection results for heatmap
        self.face_location_history = []
        
        # Create a menu
        self.menu_options = {
            '1': ('Toggle Raw Camera', self._toggle_raw_camera),
            '2': ('Toggle Grayscale', self._toggle_grayscale),
            '3': ('Toggle Thresholded', self._toggle_thresholded),
            '4': ('Toggle Face Detection', self._toggle_face_detection),
            '5': ('Toggle Eye Detection', self._toggle_eye_detection),
            '6': ('Toggle Blink Detection', self._toggle_blink_detection),
            '7': ('Toggle Glasses Detection', self._toggle_glasses_detection),
            '8': ('Toggle Stats Display', self._toggle_stats),
            '9': ('Reset History', self._reset_history),
            '0': ('Reset Blink Count', self._reset_blink_count),
            'q': ('Quit', None)
        }
    
    def _on_threshold_change(self, value):
        """Callback for threshold trackbar."""
        self.detection_threshold = value
    
    def _on_face_scale_change(self, value):
        """Callback for face scale factor trackbar."""
        self.face_scale_factor = max(1.01, value / 10.0)
    
    def _on_face_minneigh_change(self, value):
        """Callback for face min neighbors trackbar."""
        self.face_min_neighbors = max(1, value)
    
    def _on_eye_scale_change(self, value):
        """Callback for eye scale factor trackbar."""
        self.eye_scale_factor = max(1.01, value / 10.0)
    
    def _on_eye_minneigh_change(self, value):
        """Callback for eye min neighbors trackbar."""
        self.eye_min_neighbors = max(1, value)
        
    def _on_blink_duration_change(self, value):
        """Callback for blink duration threshold trackbar."""
        self.blink_duration_threshold = max(0.1, value / 10.0)
        
    def _on_looking_away_threshold_change(self, value):
        """Callback for looking away threshold trackbar."""
        self.looking_away_threshold = max(1.0, float(value))
    
    def _toggle_raw_camera(self):
        """Toggle raw camera display."""
        self.show_raw_camera = not self.show_raw_camera
    
    def _toggle_grayscale(self):
        """Toggle grayscale display."""
        self.show_grayscale = not self.show_grayscale
    
    def _toggle_thresholded(self):
        """Toggle thresholded display."""
        self.show_thresholded = not self.show_thresholded
    
    def _toggle_face_detection(self):
        """Toggle face detection visualization."""
        self.show_face_detection = not self.show_face_detection
    
    def _toggle_eye_detection(self):
        """Toggle eye detection visualization."""
        self.show_eye_detection = not self.show_eye_detection
        
    def _toggle_blink_detection(self):
        """Toggle blink detection visualization."""
        self.show_blink_detection = not self.show_blink_detection
        
    def _toggle_glasses_detection(self):
        """Toggle glasses detection visualization."""
        self.show_glasses_detection = not self.show_glasses_detection
    
    def _toggle_stats(self):
        """Toggle statistics display."""
        self.show_stats = not self.show_stats
    
    def _reset_history(self):
        """Reset face detection history."""
        self.face_location_history = []
        self.face_detected_history = deque([False] * self.face_detected_history.maxlen, maxlen=self.face_detected_history.maxlen)
        self.looking_away_history = deque([False] * self.looking_away_history.maxlen, maxlen=self.looking_away_history.maxlen)
        self.blink_history = deque([False] * self.blink_history.maxlen, maxlen=self.blink_history.maxlen)
        
    def _reset_blink_count(self):
        """Reset blink counter."""
        self.blink_count = 0
    
    def _draw_debug_info(self, frame, face_detected, faces, gray):
        """Draw debug information on the frame."""
        h, w = frame.shape[:2]
        debug_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Add different visualizations based on selected options
        if self.show_raw_camera:
            debug_frame = frame.copy()
        
        # Add grayscale visualization
        if self.show_grayscale:
            gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if self.show_raw_camera:
                # Blend with original
                debug_frame = cv2.addWeighted(debug_frame, 0.7, gray_colored, 0.3, 0)
            else:
                debug_frame = gray_colored
        
        # Add thresholded visualization
        if self.show_thresholded:
            _, thresh = cv2.threshold(gray, self.detection_threshold, 255, cv2.THRESH_BINARY)
            thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            if self.show_raw_camera or self.show_grayscale:
                debug_frame = cv2.addWeighted(debug_frame, 0.7, thresh_colored, 0.3, 0)
            else:
                debug_frame = thresh_colored
        
        # Draw face detection rectangles
        if self.show_face_detection:
            # Draw historical face detections as a heatmap
            overlay = debug_frame.copy()
            for i, (x, y, w, h) in enumerate(self.face_location_history):
                # Make more recent detections more opaque
                alpha = 0.1 * (i / len(self.face_location_history)) if len(self.face_location_history) > 0 else 0.1
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 165, 255), -1)
            
            # Apply the overlay
            debug_frame = cv2.addWeighted(overlay, 0.3, debug_frame, 0.7, 0)
            
            # Draw current face detection
            for (x, y, w, h) in faces:
                # Determine if looking away
                is_looking_away = self._is_looking_away(gray, (x, y, w, h))
                
                # Draw face rectangle
                color = (255, 0, 0) if not is_looking_away else (0, 0, 255)  # Blue if present, Red if looking away
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), color, 2)
                
                # Label if looking away
                if is_looking_away:
                    cv2.putText(debug_frame, "Looking away", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw eye regions and detected eyes if enabled
                if self.show_eye_detection:
                    # Calculate the upper region of the face used for eye detection
                    eye_y = y + int(h * 0.15)
                    eye_h = int(h * 0.4)
                    
                    # Draw eye detection region
                    cv2.rectangle(debug_frame, (x, eye_y), (x+w, eye_y+eye_h), (0, 255, 255), 1)
                    
                    # Get detected eyes
                    eye_detected, eyes, is_blinking = self._detect_eyes(gray, (x, y, w, h))
                    
                    # Draw detected eyes
                    for (ex, ey, ew, eh) in eyes:
                        # Adjust eye coordinates to be relative to the full frame, not the ROI
                        frame_ex = x + ex
                        frame_ey = eye_y + ey
                        cv2.rectangle(debug_frame, (frame_ex, frame_ey), (frame_ex+ew, frame_ey+eh), (0, 255, 0), 2)
                    
                    # Show blink detection if enabled
                    if self.show_blink_detection and is_blinking:
                        cv2.putText(debug_frame, "Blinking", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        
                        # Visualize blink history
                        blink_bar_x = x
                        blink_bar_y = y - 50
                        blink_bar_width = w
                        blink_bar_height = 10
                        
                        # Draw blink history background
                        cv2.rectangle(debug_frame, 
                                     (blink_bar_x, blink_bar_y), 
                                     (blink_bar_x + blink_bar_width, blink_bar_y + blink_bar_height), 
                                     (100, 100, 100), -1)
                        
                        # Draw blink history
                        segment_width = blink_bar_width / len(self.blink_history)
                        for i, is_blink in enumerate(self.blink_history):
                            segment_color = (255, 0, 255) if is_blink else (100, 100, 100)
                            start_x = int(blink_bar_x + i * segment_width)
                            end_x = int(start_x + segment_width)
                            if is_blink:
                                cv2.rectangle(debug_frame, 
                                             (start_x, blink_bar_y), 
                                             (end_x, blink_bar_y + blink_bar_height), 
                                             segment_color, -1)
        
        # Draw stats
        if self.show_stats:
            # Status text
            if not self.afk_status:
                status_text = "Status: Present"
                status_color = (0, 255, 0)  # Green
            else:
                status_text = f"Status: AFK - {self._get_afk_reason()}"
                status_color = (0, 0, 255)  # Red
                
            cv2.putText(debug_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Time without face
            if not face_detected:
                time_without_face = time.time() - self.last_face_time
                cv2.putText(debug_frame, f"No face: {time_without_face:.1f}s", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Looking away status
            if self.looking_away and face_detected:
                looking_away_time = time.time() - self.looking_away_start if self.looking_away_start else 0
                cv2.putText(debug_frame, f"Looking away: {looking_away_time:.1f}s / {self.looking_away_threshold:.1f}s", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Glasses detection status
            if self.show_glasses_detection:
                glasses_text = "Glasses: Detected" if self.wearing_glasses else "Glasses: Not Detected"
                glasses_confidence = self.glasses_detection_count / (self.glasses_detection_count + self.normal_eye_detection_count) * 100 if (self.glasses_detection_count + self.normal_eye_detection_count) > 0 else 0
                cv2.putText(debug_frame, f"{glasses_text} ({glasses_confidence:.1f}%)", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Blink info
            if self.show_blink_detection:
                blink_rate = self.blink_count / ((time.time() - self.last_blink_time) / 60) if (time.time() - self.last_blink_time) > 0 else 0
                cv2.putText(debug_frame, f"Blinks: {self.blink_count} (Rate: {blink_rate:.1f}/min)", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Total AFK time
            current_afk = time.time() - self.afk_start_time if self.afk_status else 0
            total_time = self.total_afk_time + current_afk
            cv2.putText(debug_frame, f"Total AFK: {datetime.timedelta(seconds=int(total_time))}", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Detection parameters
            param_y = 210
            cv2.putText(debug_frame, f"Face Scale: {self.face_scale_factor:.2f} | MinNeigh: {self.face_min_neighbors}", 
                       (10, param_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(debug_frame, f"Eye Scale: {self.eye_scale_factor:.2f} | MinNeigh: {self.eye_min_neighbors}", 
                       (10, param_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(debug_frame, f"Blink Duration: {self.blink_duration_threshold:.1f}s | Looking Away: {self.looking_away_threshold:.1f}s", 
                       (10, param_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Face detection confidence
            confidence = sum(self.face_detected_history) / len(self.face_detected_history) * 100
            cv2.putText(debug_frame, f"Face confidence: {confidence:.1f}%", 
                       (10, param_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Looking away confidence
            looking_away_confidence = sum(self.looking_away_history) / len(self.looking_away_history) * 100
            cv2.putText(debug_frame, f"Looking away confidence: {looking_away_confidence:.1f}%", 
                       (10, param_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Blink confidence
            blink_confidence = sum(self.blink_history) / len(self.blink_history) * 100
            cv2.putText(debug_frame, f"Blink confidence: {blink_confidence:.1f}%", 
                       (10, param_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Menu help
            menu_y = h - 20
            cv2.putText(debug_frame, "Menu: 0-9 to toggle options, q to quit", 
                       (10, menu_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return debug_frame
    
    def _draw_menu(self, frame):
        """Draw the menu overlay on the frame."""
        h, w = frame.shape[:2]
        menu_frame = frame.copy()
        
        # Draw a semi-transparent background for the menu
        overlay = menu_frame.copy()
        cv2.rectangle(overlay, (w-250, 10), (w-10, 320), (0, 0, 0), -1)
        menu_frame = cv2.addWeighted(overlay, 0.5, menu_frame, 0.5, 0)
        
        # Draw menu title
        cv2.putText(menu_frame, "Debug Options", (w-240, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw menu options
        y_pos = 60
        for key, (label, _) in self.menu_options.items():
            # Highlight active options
            color = (255, 255, 255)  # default white
            if key == '1' and self.show_raw_camera: color = (0, 255, 0)
            elif key == '2' and self.show_grayscale: color = (0, 255, 0)
            elif key == '3' and self.show_thresholded: color = (0, 255, 0)
            elif key == '4' and self.show_face_detection: color = (0, 255, 0)
            elif key == '5' and self.show_eye_detection: color = (0, 255, 0)
            elif key == '6' and self.show_blink_detection: color = (0, 255, 0)
            elif key == '7' and self.show_glasses_detection: color = (0, 255, 0)
            elif key == '8' and self.show_stats: color = (0, 255, 0)
            
            cv2.putText(menu_frame, f"{key}: {label}", (w-240, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 25
        
        # Draw trackbar instructions
        cv2.putText(menu_frame, "Adjust thresholds using", (w-240, y_pos+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(menu_frame, "slider controls at top", (w-240, y_pos+50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return menu_frame
    
    def run(self):
        """Run the AFK detector with debug options."""
        try:
            while True:
                # Read a frame from the camera
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera. Exiting...")
                    break
                
                # Detect face
                face_detected, faces, gray = self._detect_face(frame)
                
                # Update AFK status
                self._update_afk_status(face_detected)
                
                # Draw debug information
                debug_frame = self._draw_debug_info(frame, face_detected, faces, gray)
                
                # Draw menu
                final_frame = self._draw_menu(debug_frame)
                
                # Display the frame
                cv2.imshow('AFK Detector Debug', final_frame)
                
                # Handle key presses for menu
                key = cv2.waitKey(1) & 0xFF
                key_char = chr(key) if key > 0 and key < 256 else None
                
                if key_char in self.menu_options:
                    if key_char == 'q':
                        break
                    else:
                        # Call the associated function
                        func = self.menu_options[key_char][1]
                        if func:
                            func()
        
        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Print final stats
            print("\nAFK Sessions:")
            for i, session in enumerate(self.afk_sessions):
                start_time = datetime.datetime.fromtimestamp(session['start']).strftime('%H:%M:%S')
                end_time = datetime.datetime.fromtimestamp(session['end']).strftime('%H:%M:%S')
                print(f"  {i+1}. {start_time} - {end_time} ({session['duration']:.1f}s)")
            
            print(f"\nTotal AFK time: {datetime.timedelta(seconds=int(self.total_afk_time))}")

def main():
    parser = argparse.ArgumentParser(description='AFK Tracker Debug Menu - Visualize detection process')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    parser.add_argument('--threshold', type=float, default=3.0, 
                        help='Time in seconds without face detection to consider AFK (default: 3.0)')
    parser.add_argument('--history', type=int, default=30,
                        help='Number of frames to use for detection smoothing (default: 30)')
    
    args = parser.parse_args()
    
    # Check if Haar cascade files exist, if not download them
    cascade_files = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_eye.xml',
        cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
    ]
    
    missing_files = [f for f in cascade_files if not os.path.exists(f)]
    if missing_files:
        print("Some required model files are missing. Attempting to use OpenCV's built-in cascades.")
    
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
    print("  6: Toggle Blink Detection")
    print("  7: Toggle Glasses Detection")
    print("  8: Toggle Stats Display")
    print("  9: Reset History")
    print("  0: Reset Blink Counter")
    print("  Use sliders to adjust detection parameters")
    print("  q: Quit")
    
    detector.run()

if __name__ == "__main__":
    main() 