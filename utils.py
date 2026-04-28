"""Utility tools for the Face Recognition Attendance System"""

import logging
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import cv2

from face_encoder import FaceEncoder
from face_recognizer import FaceRecognizer
from attendance_system import AttendanceSystem
from config import (
    DATASET_DIR, ENCODINGS_FILE, LABELS_FILE, PERSON_MAPPING_FILE,
    ATTENDANCE_DIR, LOGS_DIR, ALLOWED_EXTENSIONS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "utils.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def validate_dataset():
    """Validate dataset structure and content"""
    logger.info("Validating dataset...")
    
    if not DATASET_DIR.exists():
        logger.error(f"Dataset directory not found: {DATASET_DIR}")
        return False
    
    person_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    
    if not person_dirs:
        logger.error("No person directories found")
        return False
    
    logger.info(f"Found {len(person_dirs)} person directories\n")
    
    total_images = 0
    valid_persons = 0
    
    for person_dir in person_dirs:
        image_files = [f for f in person_dir.iterdir() 
                      if f.suffix.lower() in ALLOWED_EXTENSIONS]
        
        if len(image_files) == 0:
            logger.warning(f"  {person_dir.name}: No images found")
        else:
            logger.info(f"  {person_dir.name}: {len(image_files)} images")
            total_images += len(image_files)
            valid_persons += 1
    
    logger.info(f"\nTotal persons: {valid_persons}")
    logger.info(f"Total images: {total_images}")
    
    return valid_persons > 0


def test_face_detection():
    """Test face detection on dataset"""
    logger.info("Testing face detection...\n")
    
    encoder = FaceEncoder()
    person_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    
    detection_stats = {
        'total_images': 0,
        'images_with_faces': 0,
        'total_faces': 0,
        'persons': {}
    }
    
    for person_dir in person_dirs[:3]:  # Test first 3 persons
        logger.info(f"Testing {person_dir.name}...")
        person_images = [f for f in person_dir.iterdir() 
                        if f.suffix.lower() in ALLOWED_EXTENSIONS]
        
        person_stats = {
            'images': 0,
            'with_faces': 0,
            'total_faces': 0
        }
        
        for image_file in person_images[:3]:  # Test first 3 images per person
            image, success = encoder.load_image(image_file)
            if not success:
                continue
            
            face_locations, success = encoder.detect_face(image)
            person_stats['images'] += 1
            detection_stats['total_images'] += 1
            
            if success:
                person_stats['with_faces'] += 1
                person_stats['total_faces'] += len(face_locations)
                detection_stats['images_with_faces'] += 1
                detection_stats['total_faces'] += len(face_locations)
                logger.info(f"  {image_file.name}: {len(face_locations)} face(s) detected")
            else:
                logger.warning(f"  {image_file.name}: No faces detected")
        
        detection_stats['persons'][person_dir.name] = person_stats
    
    logger.info(f"\nDetection Statistics:")
    logger.info(f"  Total images tested: {detection_stats['total_images']}")
    logger.info(f"  Images with faces: {detection_stats['images_with_faces']}")
    logger.info(f"  Total faces detected: {detection_stats['total_faces']}")
    
    return True


def display_encodings_info():
    """Display information about trained encodings"""
    if not ENCODINGS_FILE.exists() or not LABELS_FILE.exists():
        logger.error("Encodings not found. Please run train.py first.")
        return False
    
    try:
        encodings = np.load(ENCODINGS_FILE)
        labels = np.load(LABELS_FILE)
        
        with open(PERSON_MAPPING_FILE, 'r') as f:
            person_mapping = json.load(f)
        
        logger.info("\nEncoding Information:")
        logger.info(f"  Total encodings: {len(encodings)}")
        logger.info(f"  Encoding dimensions: {encodings.shape}")
        logger.info(f"  Total persons: {len(person_mapping)}")
        
        logger.info("\nPerson Details:")
        for person_id, info in person_mapping.items():
            count = np.sum(labels == int(person_id))
            logger.info(f"  {info['name']} (ID: {person_id}): {count} encodings from {info['image_count']} images")
        
        return True
    except Exception as e:
        logger.error(f"Error reading encodings: {e}")
        return False


def display_attendance_records():
    """Display all attendance records"""
    attendance_files = sorted(ATTENDANCE_DIR.glob("attendance_*.csv"))
    
    if not attendance_files:
        logger.info("No attendance records found")
        return True
    
    logger.info(f"\nFound {len(attendance_files)} attendance file(s):")
    
    for attendance_file in attendance_files:
        logger.info(f"\n  {attendance_file.name}:")
        
        try:
            with open(attendance_file, 'r') as f:
                lines = f.readlines()
                logger.info(f"    Records: {len(lines) - 1}")  # Exclude header
                
                # Show first few records
                for line in lines[:4]:
                    logger.info(f"      {line.strip()}")
                
                if len(lines) > 4:
                    logger.info(f"      ... and {len(lines) - 4} more records")
        except Exception as e:
            logger.error(f"    Error reading file: {e}")
    
    return True


def test_recognition():
    """Test face recognition"""
    if not ENCODINGS_FILE.exists():
        logger.error("Encodings not found. Please run train.py first.")
        return False
    
    try:
        logger.info("Loading encodings...")
        encodings = np.load(ENCODINGS_FILE)
        labels = np.load(LABELS_FILE)
        
        with open(PERSON_MAPPING_FILE, 'r') as f:
            person_mapping = json.load(f)
        
        recognizer = FaceRecognizer(encodings, labels, person_mapping)
        stats = recognizer.get_stats()
        
        logger.info("\nRecognizer Statistics:")
        logger.info(f"  Total encodings: {stats['total_encodings']}")
        logger.info(f"  Total persons: {stats['total_persons']}")
        logger.info(f"  Distance threshold: {stats['distance_threshold']}")
        logger.info(f"  Status: ✓ Ready for recognition")
        
        return True
    except Exception as e:
        logger.error(f"Error testing recognition: {e}")
        return False


def cleanup_logs():
    """Clean up old log files"""
    import glob
    from datetime import datetime, timedelta
    
    logger.info("Cleaning up old log files...")
    
    log_files = list(LOGS_DIR.glob("*.log"))
    removed_count = 0
    
    for log_file in log_files:
        # Keep logs from last 7 days
        if datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime) > timedelta(days=7):
            try:
                log_file.unlink()
                removed_count += 1
                logger.info(f"Removed: {log_file.name}")
            except Exception as e:
                logger.error(f"Could not remove {log_file.name}: {e}")
    
    logger.info(f"Removed {removed_count} old log files")
    return True


def main():
    """Main utility function"""
    parser = argparse.ArgumentParser(
        description="Utility tools for Face Recognition Attendance System"
    )
    parser.add_argument(
        "action",
        choices=["validate", "test-detection", "test-recognition", 
                "show-encodings", "show-attendance", "cleanup-logs"],
        help="Action to perform"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info(f"Face Recognition Attendance System - Utilities")
    logger.info("=" * 70)
    
    if args.action == "validate":
        validate_dataset()
    elif args.action == "test-detection":
        test_face_detection()
    elif args.action == "test-recognition":
        test_recognition()
    elif args.action == "show-encodings":
        display_encodings_info()
    elif args.action == "show-attendance":
        display_attendance_records()
    elif args.action == "cleanup-logs":
        cleanup_logs()
    
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
