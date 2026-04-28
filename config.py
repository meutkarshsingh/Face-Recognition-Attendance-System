"""Configuration settings for Face Recognition Attendance System"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "dataset"
ENCODINGS_DIR = PROJECT_ROOT / "encodings"
ATTENDANCE_DIR = PROJECT_ROOT / "attendance_records"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create necessary directories
ENCODINGS_DIR.mkdir(exist_ok=True)
ATTENDANCE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# File paths
ENCODINGS_FILE = ENCODINGS_DIR / "face_encodings.npy"
LABELS_FILE = ENCODINGS_DIR / "labels.npy"
PERSON_MAPPING_FILE = ENCODINGS_DIR / "person_mapping.json"

# ============================================================================
# DEEP LEARNING FACE RECOGNITION SETTINGS
# ============================================================================

# Face detection model
# - "hog": Faster, works on CPU (recommended for most cases)
# - "cnn": More accurate, requires GPU for reasonable speed
FACE_DETECTION_MODEL = "hog"

# Face recognition distance threshold
# Lower = stricter matching (fewer false positives, more false negatives)
# Higher = looser matching (more false positives, fewer false negatives)
# Recommended range: 0.4 - 0.6
DISTANCE_THRESHOLD = 0.5

# Number of jitters for face encoding
# More jitters = more accurate but slower
# 1 = fast, 5 = balanced, 10+ = very accurate but slow
NUM_JITTERS = 5

# Face encoding model size
# - "small": 5 landmarks, faster
# - "large": 68 landmarks, more accurate (recommended)
ENCODING_MODEL = "large"

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
MIN_FACE_SIZE = 20  # Minimum face size in pixels to process

# Image preprocessing
RESIZE_WIDTH = 0  # Set to > 0 to resize images for faster processing (0 = no resize)
EQUALIZE_HISTOGRAM = True  # Apply histogram equalization for lighting normalization

# ============================================================================
# DATA AUGMENTATION (for training)
# ============================================================================

USE_AUGMENTATION = True
AUGMENTATION_SETTINGS = {
    "brightness_range": (0.8, 1.2),  # Brightness variation
    "contrast_range": (0.9, 1.1),    # Contrast variation
    "horizontal_flip": True,          # Mirror images
    "rotation_range": 10,             # Rotation in degrees
}

# ============================================================================
# RECOGNITION SETTINGS
# ============================================================================

# Use k-nearest neighbors voting for more robust recognition
USE_KNN_VOTING = True
KNN_NEIGHBORS = 3  # Number of neighbors to consider

# Confidence thresholds
MIN_CONFIDENCE = 0.5  # Minimum confidence to mark as recognized
HIGH_CONFIDENCE = 0.7  # High confidence threshold

# ============================================================================
# ATTENDANCE
# ============================================================================

ATTENDANCE_FILE_PATTERN = "attendance_{date}.csv"
TIMEZONE = "Asia/Kolkata"

# Prevent duplicate attendance within this time window (seconds)
DUPLICATE_ATTENDANCE_WINDOW = 300  # 5 minutes

# ============================================================================
# MODEL VALIDATION
# ============================================================================

MIN_IMAGES_PER_PERSON = 3  # Minimum images required to train a person
TEST_SIZE = 0.2  # Proportion of data to use for testing

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"
