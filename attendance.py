"""Main attendance system - marks attendance from camera or image"""

import logging
import argparse
import sys
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from face_encoder import FaceEncoder
from face_recognizer import FaceRecognizer
from attendance_system import AttendanceSystem
from config import LOGS_DIR, ENCODINGS_FILE, LABELS_FILE, PERSON_MAPPING_FILE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "attendance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AttendanceMarker:
    """Main system to mark attendance"""

    def __init__(self):
        self.recognizer = None
        self.attendance_system = AttendanceSystem()
        self._load_encodings()

    def _load_encodings(self) -> bool:
        """Load pre-trained encodings"""
        try:
            if not ENCODINGS_FILE.exists() or not LABELS_FILE.exists():
                logger.error("Encodings not found. Please run train.py first.")
                return False
            
            encodings = np.load(ENCODINGS_FILE)
            labels = np.load(LABELS_FILE)
            
            with open(PERSON_MAPPING_FILE, 'r') as f:
                person_mapping = json.load(f)
            
            self.recognizer = FaceRecognizer(encodings, labels, person_mapping)
            logger.info("Encodings loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading encodings: {e}")
            return False

    def mark_from_image(self, image_path: str) -> bool:
        """
        Mark attendance from an image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if any faces were recognized
        """
        if self.recognizer is None:
            logger.error("Recognizer not initialized")
            return False
        
        logger.info(f"Processing image: {image_path}")
        
        results, image, success = self.recognizer.recognize_from_file(image_path)
        
        if not success:
            logger.warning("No faces detected in image")
            return False
        
        recognized_count = 0
        for result in results:
            if result["person_id"] is not None:
                person_id = str(result["person_id"])
                person_name = result["person_name"]
                confidence = result["confidence"]
                
                self.attendance_system.mark_attendance(
                    person_id, person_name, confidence, result["location"]
                )
                recognized_count += 1
                logger.info(f"✓ Recognized: {person_name} (ID: {person_id}, Confidence: {confidence:.2%})")
            else:
                logger.info(f"✗ Unknown person (Distance: {result['confidence']:.2f})")
        
        return recognized_count > 0

    def mark_from_camera(self, camera_id: int = 0, display: bool = True) -> bool:
        """
        Mark attendance from camera feed
        
        Args:
            camera_id: Camera index (default: 0 for default camera)
            display: Whether to display preview (default: True)
            
        Returns:
            True if any faces were recognized
        """
        if self.recognizer is None:
            logger.error("Recognizer not initialized")
            return False
        
        logger.info(f"Starting camera (ID: {camera_id})...")
        logger.info("Press 'q' to quit, 's' to save frame, 'r' to reset attendance")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_id}")
            return False
        
        any_recognized = False
        frame_count = 0
        skip_frames = 5  # Process every 5th frame for speed
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to grab frame")
                    break
                
                frame_count += 1
                
                # Process every nth frame
                if frame_count % skip_frames != 0:
                    if display:
                        cv2.imshow('Face Recognition Attendance', frame)
                    continue
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Recognize faces
                results, success = self.recognizer.recognize_faces_in_image(rgb_frame)
                
                if success:
                    for result in results:
                        if result["person_id"] is not None:
                            person_id = str(result["person_id"])
                            person_name = result["person_name"]
                            confidence = result["confidence"]
                            
                            # Mark attendance only once per person per session
                            if not self.attendance_system.is_person_present_today(person_id):
                                self.attendance_system.mark_attendance(
                                    person_id, person_name, confidence, result["location"]
                                )
                                logger.info(f"✓ Attendance marked: {person_name} (Confidence: {confidence:.2%})")
                                any_recognized = True
                
                # Draw results
                if display:
                    frame_with_results = self.recognizer.draw_results(rgb_frame, results)
                    cv2.imshow('Face Recognition Attendance', frame_with_results)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quitting...")
                    break
                elif key == ord('s'):
                    # Save frame
                    filename = f"frame_{frame_count}.png"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Frame saved: {filename}")
                elif key == ord('r'):
                    # Reset attendance
                    self.attendance_system.attendance_records.clear()
                    logger.info("Attendance reset")
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
        
        return any_recognized

    def save_attendance(self) -> bool:
        """Save attendance records"""
        return self.attendance_system.save_attendance()

    def print_summary(self):
        """Print attendance summary"""
        summary = self.attendance_system.get_attendance_summary()
        
        logger.info("\n" + "=" * 70)
        logger.info("Daily Attendance Summary")
        logger.info("=" * 70)
        logger.info(f"Date: {summary['date']}")
        logger.info(f"Total Unique Persons: {summary['total_persons']}")
        logger.info(f"Total Entries: {summary['total_entries']}")
        
        if summary['persons']:
            logger.info("\nPerson Details:")
            for person_id, details in summary['persons'].items():
                logger.info(f"  {details['name']} (ID: {person_id})")
                logger.info(f"    Entries: {details['entry_count']}")
                logger.info(f"    First Entry: {details['first_entry']}")
                logger.info(f"    Last Entry: {details['last_entry']}")
                logger.info(f"    Avg Confidence: {details['avg_confidence']:.2%}")
        else:
            logger.info("\nNo attendance records yet.")
        
        logger.info("=" * 70 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Face Recognition Attendance System"
    )
    parser.add_argument(
        "mode",
        choices=["camera", "image", "summary"],
        help="Mode of operation"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file (required for 'image' mode)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera ID (default: 0)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display camera preview"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Face Recognition Attendance System")
    logger.info("=" * 70)
    
    marker = AttendanceMarker()
    
    if args.mode == "camera":
        marker.mark_from_camera(
            camera_id=args.camera,
            display=not args.no_display
        )
        marker.save_attendance()
        marker.print_summary()
        return 0
    
    elif args.mode == "image":
        if not args.image:
            logger.error("--image argument required for 'image' mode")
            return 1
        
        if marker.mark_from_image(args.image):
            marker.save_attendance()
            marker.print_summary()
            return 0
        else:
            logger.warning("No recognized faces in image")
            return 1
    
    elif args.mode == "summary":
        marker.print_summary()
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
