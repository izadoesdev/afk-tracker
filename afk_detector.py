import cv2
import numpy as np
import time
import datetime
from collections import deque

class AFKDetector:
    def __init__(self, 
                 camera_index=0, 
                 face_cascade_path='haarcascade_frontalface_default.xml',
                 eye_cascade_path='haarcascade_eye.xml',
                 afk_threshold=3.0,
                 history_size=30):
        """
        Initialize the AFK detector.
        
        Args:
            camera_index: Index of the camera to use
            face_cascade_path: Path to the face cascade XML file
            eye_cascade_path: Path to the eye cascade XML file
            afk_threshold: Number of seconds without a face to consider the user AFK
            history_size: Number of face detection results to keep in history
        """
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        
        # Load the pre-trained face and eye detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + eye_cascade_path)
        
        # Also load the eye cascade for glasses
        self.eye_glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        
        # Initialize variables
        self.last_face_time = time.time()
        self.afk_threshold = afk_threshold
        self.afk_status = False
        self.afk_start_time = None
        
        # Face detection history to smooth out detections
        self.face_detected_history = deque([False] * history_size, maxlen=history_size)
        
        # Looking away history to smooth out eye detection (increased for better stability)
        self.looking_away_history = deque([False] * 20, maxlen=20)
        
        # Blink detection history
        self.blink_history = deque([False] * 10, maxlen=10)
        self.last_blink_time = time.time()
        
        # Tracking stats
        self.afk_sessions = []
        self.total_afk_time = 0
        
        # Track looking away state
        self.looking_away = False
        self.looking_away_start = None
        self.looking_away_threshold = 8.0  # Increased to 8 seconds to consider AFK when looking away
        
        # Blink detection
        self.is_blinking = False
        self.blink_count = 0
        self.blink_duration_threshold = 0.5  # Max duration for a blink in seconds
        
        # Last successful eye detection timestamp
        self.last_eyes_detected_time = time.time()
        
        # Adjust detection parameters for better performance
        self.face_scale_factor = 1.1
        self.face_min_neighbors = 5
        self.face_min_size = (50, 50)
        
        # Lower eye scale factor for more accurate detection with glasses
        self.eye_scale_factor = 1.05  
        self.eye_min_neighbors = 2  # Lower min neighbors for better detection with glasses
        self.eye_min_size = (15, 15)  # Smaller min size to detect eyes with glasses
        
        # Glasses detection flag
        self.wearing_glasses = False
        self.glasses_detection_count = 0
        self.normal_eye_detection_count = 0

    def _detect_face(self, frame):
        """Detect faces in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram to improve detection in varying lighting
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=self.face_scale_factor, 
            minNeighbors=self.face_min_neighbors, 
            minSize=self.face_min_size
        )
        
        # If no faces found with default cascade, try with alternative parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05, 
                minNeighbors=3, 
                minSize=(30, 30)
            )
        
        return len(faces) > 0, faces, gray
    
    def _detect_eyes(self, gray, face):
        """Detect eyes within a face region."""
        x, y, w, h = face
        
        # Calculate the upper region of the face (where eyes are located)
        # This reduces false positives by focusing only on the eye region
        eye_y = y + int(h * 0.15)  # Start from 15% down from the top of face
        eye_h = int(h * 0.4)      # Use only 40% of face height for eye detection
        
        # Ensure we don't go out of bounds
        if eye_y + eye_h > gray.shape[0]:
            eye_h = gray.shape[0] - eye_y
            
        # Extract eye region
        if eye_h > 0:
            roi_gray = gray[eye_y:eye_y+eye_h, x:x+w]
        else:
            roi_gray = gray[y:y+h, x:x+w]  # Fallback to full face
        
        # Increase contrast in eye region to improve detection
        roi_gray = cv2.equalizeHist(roi_gray)
        
        # If we've detected glasses recently, prioritize the glasses cascade
        eyes = []
        eyes_with_glasses = []
        
        # Try the glasses eye cascade first if we think they're wearing glasses
        if self.wearing_glasses:
            eyes_with_glasses = self.eye_glasses_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=self.eye_scale_factor,
                minNeighbors=self.eye_min_neighbors,
                minSize=self.eye_min_size
            )
            
            # If we found eyes with the glasses cascade, use those
            if len(eyes_with_glasses) > 0:
                self.glasses_detection_count += 1
                eyes = eyes_with_glasses
            else:
                # If glasses cascade failed, try regular eye cascade
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=self.eye_scale_factor,
                    minNeighbors=self.eye_min_neighbors,
                    minSize=self.eye_min_size
                )
                if len(eyes) > 0:
                    self.normal_eye_detection_count += 1
        else:
            # Try regular eye cascade first
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=self.eye_scale_factor,
                minNeighbors=self.eye_min_neighbors,
                minSize=self.eye_min_size
            )
            
            # If no eyes detected, try the glasses eye cascade
            if len(eyes) == 0:
                eyes_with_glasses = self.eye_glasses_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=self.eye_scale_factor,
                    minNeighbors=self.eye_min_neighbors,
                    minSize=self.eye_min_size
                )
                
                if len(eyes_with_glasses) > 0:
                    self.glasses_detection_count += 1
                    eyes = eyes_with_glasses
                else:
                    # Try one more time with even lower parameters
                    eyes = self.eye_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.03,
                        minNeighbors=1,
                        minSize=(10, 10)
                    )
                    if len(eyes) > 0:
                        self.normal_eye_detection_count += 1
        
        # Update glasses detection state every 50 frames
        detection_interval = 50
        if (self.glasses_detection_count + self.normal_eye_detection_count) >= detection_interval:
            self.wearing_glasses = self.glasses_detection_count > self.normal_eye_detection_count
            self.glasses_detection_count = 0
            self.normal_eye_detection_count = 0
        
        eyes_detected = len(eyes) > 0
        
        # Update last successful eye detection time
        if eyes_detected:
            self.last_eyes_detected_time = time.time()
            
        # Detect if blinking (brief loss of eye detection)
        current_time = time.time()
        if not eyes_detected and (current_time - self.last_eyes_detected_time) < self.blink_duration_threshold:
            # Likely a blink (eyes were recently detected but now briefly not detected)
            is_blink = True
        else:
            is_blink = False
        
        self.blink_history.append(is_blink)
        
        # Count blinks - pattern: eyes visible → not visible (briefly) → visible again
        if (not self.is_blinking and is_blink and 
                (current_time - self.last_blink_time > self.blink_duration_threshold)):
            self.is_blinking = True
            self.blink_count += 1
            self.last_blink_time = current_time
        elif self.is_blinking and not is_blink:
            self.is_blinking = False
            
        return eyes_detected, eyes, is_blink
    
    def _is_looking_away(self, gray, face):
        """Determine if the person is looking away based on eye detection."""
        eyes_detected, _, is_blink = self._detect_eyes(gray, face)
        
        # Add to history buffer
        # Only count as looking away if not a blink
        if is_blink:
            # If blinking, don't count as looking away
            self.looking_away_history.append(False)
        else:
            self.looking_away_history.append(not eyes_detected)
        
        # Consider looking away if a majority of recent frames show no eyes
        # Increased threshold to avoid false positives
        looking_away_ratio = sum(self.looking_away_history) / len(self.looking_away_history)
        is_looking_away = looking_away_ratio > 0.8  # 80% of recent frames show no eyes
        
        current_time = time.time()
        
        # State transitions for looking away
        if is_looking_away and not self.looking_away:
            # Just started looking away
            self.looking_away = True
            self.looking_away_start = current_time
        elif not is_looking_away and self.looking_away:
            # Just stopped looking away
            self.looking_away = False
            self.looking_away_start = None
        
        return is_looking_away
    
    def _smooth_face_detection(self, face_detected):
        """Smooth face detection to avoid flickering."""
        self.face_detected_history.append(face_detected)
        # Consider face detected if at least 1/3 of recent frames had a face
        return sum(self.face_detected_history) >= len(self.face_detected_history) // 3
    
    def _update_afk_status(self, face_detected):
        """Update AFK status based on face detection and looking away."""
        smoothed_face_detected = self._smooth_face_detection(face_detected)
        
        current_time = time.time()
        
        # AFK due to face not detected
        if not smoothed_face_detected and not self.afk_status:
            # If we've passed the threshold without seeing a face
            if (current_time - self.last_face_time) > self.afk_threshold:
                self.afk_status = True
                self.afk_start_time = self.last_face_time + self.afk_threshold
                print(f"You are now AFK (no face detected) - started at {datetime.datetime.fromtimestamp(self.afk_start_time).strftime('%H:%M:%S')}")
        
        # AFK due to looking away for too long
        elif self.looking_away and not self.afk_status:
            # If we've been looking away for longer than the threshold
            if self.looking_away_start and (current_time - self.looking_away_start) > self.looking_away_threshold:
                self.afk_status = True
                self.afk_start_time = self.looking_away_start + self.looking_away_threshold
                print(f"You are now AFK (looking away) - started at {datetime.datetime.fromtimestamp(self.afk_start_time).strftime('%H:%M:%S')}")
        
        # Return from AFK status
        elif smoothed_face_detected and self.afk_status and not self.looking_away:
            # We're back!
            afk_duration = current_time - self.afk_start_time
            self.afk_sessions.append({
                'start': self.afk_start_time,
                'end': current_time,
                'duration': afk_duration
            })
            self.total_afk_time += afk_duration
            self.afk_status = False
            print(f"Welcome back! You were AFK for {afk_duration:.1f} seconds")
        
        # Update last_face_time if face is detected
        if smoothed_face_detected:
            self.last_face_time = current_time
    
    def _get_afk_reason(self):
        """Return the reason for being AFK."""
        if not self.afk_status:
            return "Present"
        
        current_time = time.time()
        
        # Determine if AFK due to no face or looking away
        if self.looking_away and self.looking_away_start:
            return f"Looking Away ({current_time - self.afk_start_time:.1f}s)"
        else:
            return f"No Face ({current_time - self.afk_start_time:.1f}s)"
    
    def _draw_info(self, frame, face_detected, faces, gray):
        """Draw information on the frame."""
        # Draw status text
        if not self.afk_status:
            status_text = "Status: Present"
            status_color = (0, 255, 0)  # Green
        else:
            status_text = f"Status: AFK - {self._get_afk_reason()}"
            status_color = (0, 0, 255)  # Red
            
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw time without face if no face is detected
        if not face_detected:
            time_without_face = time.time() - self.last_face_time
            cv2.putText(frame, f"No face: {time_without_face:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw looking away status
        if self.looking_away and face_detected:
            looking_away_time = time.time() - self.looking_away_start if self.looking_away_start else 0
            cv2.putText(frame, f"Looking away: {looking_away_time:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Draw glasses detection status
        glasses_text = "Glasses: Detected" if self.wearing_glasses else "Glasses: Not Detected"
        cv2.putText(frame, glasses_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw blink info
        cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Draw total AFK time
        current_afk = time.time() - self.afk_start_time if self.afk_status else 0
        total_time = self.total_afk_time + current_afk
        y_pos = 180
        cv2.putText(frame, f"Total AFK: {datetime.timedelta(seconds=int(total_time))}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw rectangles around faces and check for looking away
        for (x, y, w, h) in faces:
            # Determine if looking away
            is_looking_away = self._is_looking_away(gray, (x, y, w, h))
            
            # Draw face rectangle
            color = (255, 0, 0) if not is_looking_away else (0, 0, 255)  # Blue if present, Red if looking away
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Label if looking away
            if is_looking_away:
                cv2.putText(frame, "Looking away", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw eye regions
            is_eyes_detected, eyes, is_blinking = self._detect_eyes(gray, (x, y, w, h))
            
            # Calculate the upper region of the face used for eye detection
            eye_y = y + int(h * 0.15)
            eye_h = int(h * 0.4)
            
            # Draw eye detection region
            cv2.rectangle(frame, (x, eye_y), (x+w, eye_y+eye_h), (0, 255, 255), 1)
            
            # Show blinking status
            if is_blinking:
                cv2.putText(frame, "Blinking", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Draw detected eyes
            for (ex, ey, ew, eh) in eyes:
                # Adjust eye coordinates to be relative to the full frame, not the ROI
                frame_ex = x + ex
                frame_ey = eye_y + ey
                cv2.rectangle(frame, (frame_ex, frame_ey), (frame_ex+ew, frame_ey+eh), (0, 255, 0), 2)
    
    def run(self):
        """Run the AFK detector."""
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
                
                # Draw information on frame
                self._draw_info(frame, face_detected, faces, gray)
                
                # Display the frame
                cv2.imshow('AFK Detector', frame)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
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
    detector = AFKDetector()
    detector.run()

if __name__ == "__main__":
    main() 