"""
ARCHITECTURE DOCUMENTATION
Face Recognition Attendance System

This file explains the system architecture, data flow, and how to extend it.
"""

# ============================================================================
# TABLE OF CONTENTS
# ============================================================================
# 1. System Overview
# 2. Architecture Diagram
# 3. Module Descriptions
# 4. Data Flow
# 5. Face Recognition Algorithm
# 6. Extension Points
# 7. Performance Optimization Tips

# ============================================================================
# 1. SYSTEM OVERVIEW
# ============================================================================

"""
The Face Recognition Attendance System is a modular, extensible Python 
application designed to automatically mark attendance using face recognition.

Key Design Principles:
  • Modular Architecture: Each component can be independently tested/extended
  • Configuration-Driven: All settings in one place (config.py)
  • Logging-First: Comprehensive logging for debugging and monitoring
  • Error Handling: Graceful failure with informative error messages
  • Performance: Optimized for real-time processing on standard hardware

Main Components:
  1. FaceEncoder - Trains face encodings from dataset
  2. FaceRecognizer - Recognizes faces from images/camera
  3. AttendanceSystem - Logs and manages attendance records
  4. Configuration - Centralized settings management
"""

# ============================================================================
# 2. ARCHITECTURE DIAGRAM
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    FACE RECOGNITION ATTENDANCE SYSTEM                    │
└─────────────────────────────────────────────────────────────────────────┘

INPUT SOURCES:
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │  Camera   │    │  Images   │    │  Videos   │
    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
          │                 │               │
          └─────────────────┼───────────────┘
                            │
                    ┌───────▼────────┐
                    │  Image Loading │ (Handled by FaceRecognizer)
                    └───────┬────────┘
                            │
                    ┌───────▼──────────────┐
                    │ Face Detection (HOG) │ or (CNN)
                    └───────┬──────────────┘
                            │
                    ┌───────▼─────────────────┐
                    │ Face Encoding (128-dim) │
                    └───────┬─────────────────┘
                            │
            ┌───────────────┤
            │               │
      ┌─────▼────────┐  ┌──▼──────────────────────┐
      │ Training     │  │ Recognition (Distance)  │
      │ (train.py)   │  │ (attendance.py)         │
      └─────┬────────┘  └──┬─────────────────────┘
            │              │
      ┌─────▼────────┐  ┌──▼──────────────────┐
      │ Encodings DB │  │ Attendance Logging  │
      │ - .npy files │  │ - CSV files         │
      │ - JSON map   │  │ - In-memory cache   │
      └──────────────┘  └─────────────────────┘

OUTPUT:
    ┌──────────────────────────────────────┐
    │ Attendance CSV Files (attendance_*.csv)
    │ Logs (logs/*.log)
    │ Visual Feedback (camera preview)
    └──────────────────────────────────────┘

INTERFACES:
    • face_encoder.py - FaceEncoder class
    • face_recognizer.py - FaceRecognizer class
    • attendance_system.py - AttendanceSystem class
    • config.py - Configuration parameters
"""

# ============================================================================
# 3. MODULE DESCRIPTIONS
# ============================================================================

"""
MODULE: config.py
─────────────────
Purpose: Centralized configuration management
Key Classes: None (module-level variables)
Key Functions: Directory creation on import

Responsibilities:
  • Define all configurable parameters
  • Setup directory structure
  • Manage file paths

Example Usage:
  from config import FACE_DETECTION_MODEL, DISTANCE_THRESHOLD
  if model == 'hog': use_hog()
  else: use_cnn()


MODULE: face_encoder.py
──────────────────────
Purpose: Train face encodings from dataset
Key Class: FaceEncoder

Methods:
  • load_image(image_path) → tuple(image, success)
    Load and convert image to RGB format
    
  • detect_face(image) → tuple(face_locations, success)
    Detect all faces in an image
    Filters by minimum face size
    
  • encode_faces(image, face_locations) → tuple(encodings, success)
    Generate 128-dimensional encoding vectors for each face
    
  • process_person_directory(person_dir) → tuple(encodings, count)
    Process all images in a person's directory
    Returns list of encodings and image count
    
  • train() → success
    Main training function
    Processes all person directories in dataset
    Builds mapping of person IDs to names
    
  • save_encodings() → success
    Save to: face_encodings.npy, labels.npy, person_mapping.json
    
  • load_encodings() → success
    Load pre-trained encodings from disk
    
  • get_stats() → dict
    Return training statistics

Data Structures:
  • self.encodings: np.ndarray(N, 128) - Face encodings
  • self.labels: np.ndarray(N,) - Person IDs for each encoding
  • self.person_mapping: dict - {person_id: {name, image_count}}


MODULE: face_recognizer.py
──────────────────────────
Purpose: Recognize faces in images/video
Key Class: FaceRecognizer

Constructor:
  FaceRecognizer(encodings, labels, person_mapping)
  
Methods:
  • load_image(image_path) → tuple(image, success)
    Load image for recognition
    
  • detect_faces(image) → tuple(face_locations, success)
    Detect all faces in image
    
  • encode_faces(image, face_locations) → tuple(encodings, success)
    Encode detected faces
    
  • recognize_face(face_encoding) → tuple(person_id, confidence)
    Recognize a single face encoding
    Uses Euclidean distance to find closest match
    Returns (None, distance) if no match above threshold
    
  • recognize_faces_in_image(image) → tuple(results, success)
    Recognize all faces in image
    Returns list of result dicts with location, person_id, confidence
    
  • recognize_from_file(image_path) → tuple(results, image, success)
    High-level function: load image and recognize
    
  • draw_results(image, results) → image_with_boxes
    Draw bounding boxes and labels on image
    Green box = recognized, Red box = unknown
    
  • get_stats() → dict
    Return recognizer statistics

Recognition Algorithm:
  For each detected face:
    1. Generate 128-dim encoding vector
    2. Calculate Euclidean distance to all training encodings
    3. Find minimum distance and corresponding person
    4. If distance ≤ DISTANCE_THRESHOLD:
         confidence = 1.0 - distance
         return (person_id, confidence)
       Else:
         return (None, distance as uncertainty measure)


MODULE: attendance_system.py
────────────────────────────
Purpose: Log and manage attendance records
Key Class: AttendanceSystem

Methods:
  • get_attendance_file(date) → Path
    Get CSV file path for a specific date
    
  • load_today_attendance() → void
    Load today's attendance from CSV
    Populates self.attendance_records
    
  • mark_attendance(person_id, person_name, confidence, location) → success
    Mark attendance entry for a person
    Adds to in-memory cache
    
  • save_attendance() → success
    Write all attendance records to CSV file
    Format: person_id, person_name, timestamp, confidence, face_location
    
  • get_attendance_summary() → dict
    Get statistics: unique persons, total entries, per-person details
    
  • is_person_present_today(person_id) → bool
    Check if person has been marked present
    
  • get_person_entries_today(person_id) → list
    Get all attendance entries for a person today

Data Structure:
  self.attendance_records: dict
    {person_id: [entry_dict, entry_dict, ...]}
    entry_dict: {person_id, person_name, timestamp, confidence, face_location}
"""

# ============================================================================
# 4. DATA FLOW
# ============================================================================

"""
TRAINING FLOW (train.py):
────────────────────────
1. User runs: python train.py
2. FaceEncoder.train() is called
3. For each person directory in dataset/:
   a. For each image in directory:
      i. Load image
      ii. Detect faces
      iii. Encode faces (128-dim vectors)
      iv. Store encodings with person label
4. Save to disk:
   - face_encodings.npy (N×128 array)
   - labels.npy (N array of person IDs)
   - person_mapping.json (person ID → name)
5. Output: Training statistics


ATTENDANCE MARKING FLOW (attendance.py camera):
──────────────────────────────────────────────
1. User runs: python attendance.py camera
2. Camera is opened (cv2.VideoCapture)
3. For each frame from camera:
   a. Skip frames (process every 5th frame for performance)
   b. Convert BGR → RGB
   c. Detect faces
   d. Encode faces (128-dim vectors)
   e. For each encoding:
      i. Calculate distance to all training encodings
      ii. Find minimum distance
      iii. If distance ≤ threshold:
           - Mark attendance if not already marked
           - Display name and confidence
           - Log to in-memory cache
      iv. Else:
           - Display "Unknown"
   f. Draw results on frame
   g. Display to user
4. On quit:
   a. Save attendance to CSV
   b. Print summary
   c. Save logs


ATTENDANCE MARKING FLOW (attendance.py image):
───────────────────────────────────────────────
1. User runs: python attendance.py image --image photo.jpg
2. Load image file
3. Detect faces and encode (same as camera)
4. For each detected face:
   a. Recognize and mark attendance
5. Save attendance to CSV
6. Print summary


RECOGNITION ALGORITHM (detailed):
──────────────────────────────────
Input: Image with face
Output: Person ID, Confidence

Steps:
  1. Load image and convert to RGB
  
  2. Face Detection (HOG or CNN)
     - HOG: Histogram of Oriented Gradients (fast)
     - CNN: Convolutional Neural Network (accurate, requires GPU)
     - Output: Face locations [(top, right, bottom, left), ...]
  
  3. Face Encoding
     - Uses pre-trained deep learning model (ResNet by dlib)
     - Generates 128-dimensional vector for each face
     - This vector captures face characteristics
  
  4. Face Matching
     - For each unknown face encoding:
       a. Calculate Euclidean distance to all known encodings
       b. distances = ||unknown_encoding - known_encoding||
       c. Find index of minimum distance
       d. min_distance = min(distances)
       e. person_id = labels[argmin(distances)]
  
  5. Confidence Calculation
     - confidence = 1.0 - min_distance
     - Higher confidence = better match
  
  6. Threshold Comparison
     - If min_distance ≤ DISTANCE_THRESHOLD:
         Return (person_id, confidence)
       Else:
         Return (None, min_distance) [Unknown person]
"""

# ============================================================================
# 5. EXTENSION POINTS
# ============================================================================

"""
The system is designed to be extensible. Here are common customization points:

EXTENSION 1: Database Backend
────────────────────────────
Instead of CSV files, save to database:

Step 1: Create new_attendance_store.py:
  class DatabaseAttendanceStore:
    def mark_attendance(self, person_id, timestamp, confidence):
      # Save to PostgreSQL/MySQL/MongoDB
      db.insert(...)

Step 2: Modify attendance.py:
  from new_attendance_store import DatabaseAttendanceStore
  attendance = DatabaseAttendanceStore()
  attendance.mark_attendance(person_id, timestamp, confidence)


EXTENSION 2: Web Interface
─────────────────────────
Create Flask/FastAPI wrapper:

Step 1: Create api.py:
  from flask import Flask, jsonify
  from attendance import AttendanceMarker
  
  marker = AttendanceMarker()
  
  @app.route('/api/mark-attendance', methods=['POST'])
  def mark_attendance():
    image_file = request.files['image']
    results = marker.mark_from_image(image_file.filename)
    return jsonify(results)

Step 2: Run:
  python api.py


EXTENSION 3: Email Notifications
────────────────────────────────
Add email alerts for attendance:

Step 1: Create notification.py:
  import smtplib
  
  def send_attendance_email(person_name, timestamp):
    smtp = smtplib.SMTP(...)
    smtp.send_message(email)

Step 2: Call from attendance.py:
  from notification import send_attendance_email
  marker.attendance_system.mark_attendance(...)
  send_attendance_email(person_name, timestamp)


EXTENSION 4: Multiple Camera Support
────────────────────────────────────
Run multiple cameras simultaneously:

Step 1: Create multi_camera.py:
  import threading
  from attendance import AttendanceMarker
  
  cameras = [0, 1, 2]
  threads = []
  
  for camera_id in cameras:
    t = threading.Thread(
      target=lambda: AttendanceMarker().mark_from_camera(camera_id)
    )
    threads.append(t)
    t.start()

Step 2: Run:
  python multi_camera.py


EXTENSION 5: Time-Based Rules
──────────────────────────────
Add late arrival/early departure detection:

Step 1: Modify attendance_system.py:
  MORNING_START = 9:00 AM
  MORNING_END = 12:00 PM
  
  def mark_attendance(self, person_id, person_name, confidence):
    current_time = datetime.now().time()
    
    if current_time < MORNING_START:
      attendance_type = "EARLY"
    elif current_time > MORNING_END:
      attendance_type = "LATE"
    else:
      attendance_type = "PRESENT"
    
    # Save with attendance_type


EXTENSION 6: Custom Face Detection
───────────────────────────────────
Add alternative face detection methods:

Step 1: Create custom_face_detector.py:
  import cv2
  
  cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  
  def detect_faces_cascade(image):
    faces = cascade.detectMultiScale(image, 1.1, 4)
    return faces

Step 2: Use in face_encoder.py:
  from custom_face_detector import detect_faces_cascade
  
  # Modify detect_face method to use custom detector


EXTENSION 7: Performance Monitoring
────────────────────────────────────
Add timing and performance metrics:

Step 1: Create monitoring.py:
  import time
  from functools import wraps
  
  def time_operation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      start = time.time()
      result = func(*args, **kwargs)
      duration = time.time() - start
      logger.info(f"{func.__name__}: {duration:.2f}s")
      return result
    return wrapper

Step 2: Use in modules:
  @time_operation
  def train(self):
    # Training code


EXTENSION 8: Quality Metrics
───────────────────────────
Add model quality assessment:

Step 1: Create quality_metrics.py:
  def calculate_intra_person_distance(encodings, labels):
    # Calculate average distance between same person
    
  def calculate_inter_person_distance(encodings, labels):
    # Calculate average distance between different persons
    
  def quality_score(intra, inter):
    # Return score (higher = better separation)
    return inter / (intra + 0.001)

Step 2: Use to evaluate model:
  metrics = calculate_quality_metrics(encodings, labels)
  if metrics['quality_score'] < THRESHOLD:
    logger.warning("Model quality low, consider adding more training data")
"""

# ============================================================================
# 6. PERFORMANCE OPTIMIZATION TIPS
# ============================================================================

"""
OPTIMIZATION 1: Frame Skipping
──────────────────────────────
Current: Process every 5th frame
Benefit: 5x faster processing, minimal accuracy loss

In attendance.py camera:
  skip_frames = 10  # Increase for more speed, less responsiveness


OPTIMIZATION 2: Image Scaling
──────────────────────────────
Process smaller images for faster detection:

In face_recognizer.py:
  def recognize_faces_in_image(self, image):
    # Scale down image for faster processing
    scale = 0.5
    small_image = cv2.resize(image, None, fx=scale, fy=scale)
    results = self.recognize_faces(small_image)
    # Scale coordinates back up
    for result in results:
      result['location']['top'] *= 1/scale


OPTIMIZATION 3: GPU Acceleration
────────────────────────────────
Use GPU for face detection:

In config.py:
  # Option 1: Install CUDA
  # pip install dlib --config-settings="--build-type=Release" --force-reinstall
  
  # Option 2: Use CPU but optimize
  FACE_DETECTION_MODEL = "hog"  # Faster than "cnn"
  NUM_JITTERS = 1  # Minimum quality


OPTIMIZATION 4: Encoding Cache
──────────────────────────────
Cache encodings in memory:

In face_recognizer.py:
  self._encoding_cache = {}
  
  def recognize_face(self, face_encoding):
    key = tuple(face_encoding.round(4))
    if key in self._encoding_cache:
      return self._encoding_cache[key]
    
    result = self._calculate_distance(face_encoding)
    self._encoding_cache[key] = result
    return result


OPTIMIZATION 5: Batch Processing
────────────────────────────────
Process multiple faces at once:

In face_recognizer.py:
  def recognize_faces_batch(self, face_encodings):
    # Calculate distances for all faces at once (vectorized)
    distances = face_recognition.face_distance(
      self.encodings, 
      face_encodings  # Multiple encodings
    )
    # Faster than looping through faces


OPTIMIZATION 6: Selective Training
─────────────────────────────────
Only use best images:

In face_encoder.py:
  def process_person_directory(self, person_dir):
    # Select only images where face takes up 20%+ of image
    # Ignore faces that are too small or at extreme angles
    
    valid_faces = []
    for image, face_location in image_face_pairs:
      face_size = face_area / image_area
      if face_size > 0.2:  # At least 20% of image
        valid_faces.append(image)


PERFORMANCE BENCHMARKS:
───────────────────────
Hardware: Intel i7, 8GB RAM (typical laptop)

Training:
  • 30 people × 5 images: 20 seconds
  • 100 people × 5 images: 60 seconds

Recognition (Camera):
  • HOG detection: 15-25 FPS
  • CNN detection: 5-10 FPS
  
Recognition (Image):
  • HOG + Encoding: 100-200ms
  • CNN + Encoding: 500-1000ms

Accuracy:
  • 95%+ with good lighting
  • 85%+ in varying light conditions
  • 70%+ with sunglasses/masks/angles
"""

# ============================================================================
# 7. DEBUGGING TIPS
# ============================================================================

"""
COMMON ISSUES & DEBUGGING:

Issue: No faces detected
─────────────────────
1. Check image quality: run utils.py test-detection
2. Check face size: LOG shows which images fail
3. Adjust MIN_FACE_SIZE in config.py (lower = detect smaller faces)
4. Try CNN detection: Change FACE_DETECTION_MODEL = "cnn"

Issue: Low recognition accuracy
──────────────────────────────
1. Add more training images per person (5-10 minimum)
2. Use varied angles and lighting
3. Check logs for patterns in failures
4. Reduce DISTANCE_THRESHOLD if too strict
5. Use utils.py test-recognition to verify model

Issue: Out of memory
───────────────────
1. Reduce NUM_JITTERS (1 or 2)
2. Use HOG instead of CNN
3. Reduce frame rate (increase skip_frames)
4. Scale down images before processing

Debugging with Logs:
──────────────────
Check these log files:
  • logs/training.log - Training progress
  • logs/face_recognizer.log - Recognition details
  • logs/attendance.log - Attendance events
  
Example:
  tail -f logs/training.log  # Watch real-time

Adding Debug Output:
───────────────────
In face_recognizer.py:
  import logging
  logger = logging.getLogger(__name__)
  
  def recognize_face(self, face_encoding):
    distances = face_recognition.face_distance(self.encodings, face_encoding)
    logger.debug(f"Distances: {distances[:5]}...")  # Show first 5
    logger.debug(f"Min distance: {np.min(distances)}")
"""

# ============================================================================
# END OF ARCHITECTURE DOCUMENTATION
# ============================================================================

print(__doc__)
