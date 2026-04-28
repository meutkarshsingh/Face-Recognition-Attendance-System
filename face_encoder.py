"""
Face encoder module using deep learning for high-accuracy face encodings.

Uses the face_recognition library (dlib) with a 99.38% accurate deep learning model
for generating 128-dimensional face embeddings.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import face_recognition

from config import (
    DATASET_DIR, ENCODINGS_FILE, LABELS_FILE, PERSON_MAPPING_FILE,
    ALLOWED_EXTENSIONS, MIN_FACE_SIZE, FACE_DETECTION_MODEL, NUM_JITTERS,
    MIN_IMAGES_PER_PERSON, LOGS_DIR, ENCODING_MODEL, USE_AUGMENTATION,
    AUGMENTATION_SETTINGS, EQUALIZE_HISTOGRAM
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "face_encoder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FaceEncoder:
    """
    Deep learning face encoder using dlib's ResNet model.
    
    Generates 128-dimensional embeddings that are highly discriminative
    for face recognition tasks.
    """

    def __init__(self):
        self.encodings = []
        self.labels = []
        self.person_mapping = {}
        self.person_id_counter = 0
        logger.info(f"FaceEncoder initialized with model={ENCODING_MODEL}, jitters={NUM_JITTERS}")

    def load_image(self, image_path: str) -> Tuple[Optional[np.ndarray], bool]:
        """
        Load an image from file in RGB format for face_recognition library.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (image array in RGB format, success flag)
        """
        try:
            # Use face_recognition's loader which handles RGB conversion
            image = face_recognition.load_image_file(str(image_path))
            
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return None, False
            
            return image, True
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None, False

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to improve face detection and encoding.
        
        Args:
            image: Image array in RGB format
            
        Returns:
            Preprocessed image
        """
        if EQUALIZE_HISTOGRAM:
            # Convert to LAB color space for better histogram equalization
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return image

    def detect_faces(self, image: np.ndarray) -> Tuple[List, bool]:
        """
        Detect faces using deep learning (CNN) or HOG model.
        
        Args:
            image: Image array in RGB format
            
        Returns:
            Tuple of (face locations list, success flag)
        """
        try:
            # Detect faces using the configured model
            face_locations = face_recognition.face_locations(
                image, 
                model=FACE_DETECTION_MODEL
            )
            
            if len(face_locations) == 0:
                return [], False
            
            # Filter by minimum face size
            filtered_locations = []
            for (top, right, bottom, left) in face_locations:
                face_height = bottom - top
                face_width = right - left
                if face_height >= MIN_FACE_SIZE and face_width >= MIN_FACE_SIZE:
                    filtered_locations.append((top, right, bottom, left))
            
            return filtered_locations, len(filtered_locations) > 0
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return [], False

    def encode_faces(self, image: np.ndarray, face_locations: List) -> Tuple[List[np.ndarray], bool]:
        """
        Generate 128-dimensional deep learning embeddings for detected faces.
        
        Uses dlib's ResNet model trained on millions of faces for
        highly discriminative face representations.
        
        Args:
            image: Image array in RGB format
            face_locations: List of face locations (top, right, bottom, left)
            
        Returns:
            Tuple of (list of 128-dim encodings, success flag)
        """
        try:
            # Generate face encodings using deep learning
            encodings = face_recognition.face_encodings(
                image,
                known_face_locations=face_locations,
                num_jitters=NUM_JITTERS,
                model=ENCODING_MODEL
            )
            
            return encodings, len(encodings) > 0
        except Exception as e:
            logger.error(f"Error encoding faces: {e}")
            return [], False

    def augment_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply data augmentation to generate variations of the image.
        
        Args:
            image: Original image in RGB format
            
        Returns:
            List of augmented images including original
        """
        augmented = [image]  # Always include original
        
        if not USE_AUGMENTATION:
            return augmented
        
        settings = AUGMENTATION_SETTINGS
        
        # Brightness variations
        if "brightness_range" in settings:
            low, high = settings["brightness_range"]
            for factor in [low, high]:
                adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
                augmented.append(adjusted)
        
        # Contrast variations
        if "contrast_range" in settings:
            low, high = settings["contrast_range"]
            for factor in [low, high]:
                mean = np.mean(image)
                adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=(1 - factor) * mean)
                augmented.append(adjusted)
        
        # Horizontal flip
        if settings.get("horizontal_flip", False):
            flipped = cv2.flip(image, 1)
            augmented.append(flipped)
        
        return augmented

    def process_person_directory(self, person_dir: Path) -> Tuple[List[np.ndarray], int]:
        """
        Process all images in a person's directory with data augmentation.
        
        Args:
            person_dir: Path to person's directory
            
        Returns:
            Tuple of (list of encodings, number of processed images)
        """
        person_name = person_dir.name
        person_encodings = []
        processed_count = 0
        
        logger.info(f"Processing person: {person_name}")
        
        # Get all image files
        image_files = [f for f in person_dir.iterdir() 
                      if f.suffix.lower() in ALLOWED_EXTENSIONS]
        
        if len(image_files) < MIN_IMAGES_PER_PERSON:
            logger.warning(f"Person {person_name} has only {len(image_files)} images. "
                          f"Minimum required: {MIN_IMAGES_PER_PERSON}")
            return [], 0
        
        for image_file in image_files:
            image, success = self.load_image(image_file)
            if not success:
                continue
            
            # Preprocess image
            image = self.preprocess_image(image)
            
            # Get augmented versions
            augmented_images = self.augment_image(image)
            
            for aug_image in augmented_images:
                face_locations, success = self.detect_faces(aug_image)
                if not success:
                    continue
                
                encodings, success = self.encode_faces(aug_image, face_locations)
                if not success:
                    continue
                
                # Use the first face encoding if multiple faces detected
                if encodings:
                    person_encodings.append(encodings[0])
            
            processed_count += 1
            logger.debug(f"Encoded face from {image_file.name} ({len(augmented_images)} variations)")
        
        logger.info(f"Processed {processed_count} images for {person_name} "
                   f"({len(person_encodings)} total encodings)")
        return person_encodings, processed_count

    def train(self) -> bool:
        """
        Train the face encoder on the dataset using deep learning embeddings.
        
        Returns:
            True if training successful, False otherwise
        """
        logger.info("=" * 70)
        logger.info("Starting DEEP LEARNING face encoding training...")
        logger.info(f"Detection model: {FACE_DETECTION_MODEL}")
        logger.info(f"Encoding model: {ENCODING_MODEL}")
        logger.info(f"Jitters: {NUM_JITTERS}")
        logger.info(f"Data augmentation: {USE_AUGMENTATION}")
        logger.info("=" * 70)
        
        if not DATASET_DIR.exists():
            logger.error(f"Dataset directory not found: {DATASET_DIR}")
            return False
        
        # Get all person directories
        person_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
        
        if not person_dirs:
            logger.error("No person directories found in dataset")
            return False
        
        total_images_processed = 0
        valid_persons = 0
        
        for person_dir in person_dirs:
            encodings, processed = self.process_person_directory(person_dir)
            
            if processed == 0:
                logger.warning(f"Skipping {person_dir.name}: No valid images")
                continue
            
            # Assign person ID
            person_id = self.person_id_counter
            self.person_mapping[person_id] = {
                "name": person_dir.name,
                "image_count": processed,
                "encoding_count": len(encodings)
            }
            self.person_id_counter += 1
            valid_persons += 1
            total_images_processed += processed
            
            # Add encodings and labels
            for encoding in encodings:
                self.encodings.append(encoding)
                self.labels.append(person_id)
        
        if total_images_processed == 0:
            logger.error("No valid encodings generated")
            return False
        
        # Convert to numpy arrays
        self.encodings = np.array(self.encodings)
        self.labels = np.array(self.labels)
        
        logger.info("=" * 70)
        logger.info("Training complete!")
        logger.info(f"Valid persons: {valid_persons}")
        logger.info(f"Total images processed: {total_images_processed}")
        logger.info(f"Total encodings generated: {len(self.encodings)}")
        logger.info(f"Avg encodings per person: {len(self.encodings) / valid_persons:.1f}")
        logger.info("=" * 70)
        
        return True

    def save_encodings(self) -> bool:
        """
        Save encodings, labels, and person mapping to disk.
        
        Returns:
            True if save successful, False otherwise
        """
        try:
            np.save(ENCODINGS_FILE, self.encodings)
            np.save(LABELS_FILE, self.labels)
            
            with open(PERSON_MAPPING_FILE, 'w') as f:
                json.dump(self.person_mapping, f, indent=4)
            
            logger.info(f"Encodings saved to {ENCODINGS_FILE}")
            logger.info(f"Labels saved to {LABELS_FILE}")
            logger.info(f"Person mapping saved to {PERSON_MAPPING_FILE}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving encodings: {e}")
            return False

    def load_encodings(self) -> bool:
        """
        Load encodings, labels, and person mapping from disk.
        
        Returns:
            True if load successful, False otherwise
        """
        try:
            if not ENCODINGS_FILE.exists() or not LABELS_FILE.exists():
                logger.error("Encoding files not found. Please train first.")
                return False
            
            self.encodings = np.load(ENCODINGS_FILE)
            self.labels = np.load(LABELS_FILE)
            
            with open(PERSON_MAPPING_FILE, 'r') as f:
                self.person_mapping = json.load(f)
            
            logger.info(f"Loaded {len(self.encodings)} encodings for {len(self.person_mapping)} persons")
            return True
        except Exception as e:
            logger.error(f"Error loading encodings: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get training statistics"""
        return {
            "total_encodings": len(self.encodings) if isinstance(self.encodings, np.ndarray) else 0,
            "total_persons": len(self.person_mapping),
            "person_mapping": self.person_mapping,
            "model": ENCODING_MODEL,
            "jitters": NUM_JITTERS,
            "detection_model": FACE_DETECTION_MODEL
        }
