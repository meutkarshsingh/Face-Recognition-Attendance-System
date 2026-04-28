"""
Face recognition module using deep learning for high-accuracy identification.

Uses the face_recognition library (dlib) with a 99.38% accurate deep learning model
and k-nearest neighbors voting for robust face matching.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import cv2
import face_recognition

from config import (
    DISTANCE_THRESHOLD, FACE_DETECTION_MODEL, NUM_JITTERS,
    LOGS_DIR, ENCODING_MODEL, USE_KNN_VOTING, KNN_NEIGHBORS,
    MIN_CONFIDENCE, HIGH_CONFIDENCE, EQUALIZE_HISTOGRAM, MIN_FACE_SIZE
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "face_recognizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Deep learning face recognizer using dlib's ResNet model.
    
    Features:
    - 128-dimensional deep learning embeddings
    - K-nearest neighbors voting for robust matching
    - Confidence scoring based on face distance
    """

    def __init__(self, encodings: np.ndarray, labels: np.ndarray, 
                 person_mapping: Dict):
        """
        Initialize the recognizer with trained encodings.
        
        Args:
            encodings: Numpy array of face encodings (N x 128)
            labels: Numpy array of person labels (N,)
            person_mapping: Dictionary mapping person IDs to person info
        """
        self.encodings = encodings
        self.labels = labels
        self.person_mapping = person_mapping
        
        # Precompute for faster matching
        self._encodings_array = np.array(encodings) if len(encodings) > 0 else np.array([])
        
        logger.info(f"FaceRecognizer initialized with {len(encodings)} encodings, "
                   f"threshold={DISTANCE_THRESHOLD}, KNN={USE_KNN_VOTING}")

    def load_image(self, image_path: str) -> Tuple[Optional[np.ndarray], bool]:
        """
        Load an image from file in RGB format.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (image array in RGB format, success flag)
        """
        try:
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
        Apply preprocessing for better recognition.
        
        Args:
            image: Image array in RGB format
            
        Returns:
            Preprocessed image
        """
        if EQUALIZE_HISTOGRAM:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image

    def detect_faces(self, image: np.ndarray) -> Tuple[List, bool]:
        """
        Detect all faces in an image using deep learning.
        
        Args:
            image: Image array in RGB format
            
        Returns:
            Tuple of (face locations list, success flag)
        """
        try:
            face_locations = face_recognition.face_locations(
                image, 
                model=FACE_DETECTION_MODEL
            )
            
            # Filter by minimum face size
            filtered = []
            for (top, right, bottom, left) in face_locations:
                if (bottom - top) >= MIN_FACE_SIZE and (right - left) >= MIN_FACE_SIZE:
                    filtered.append((top, right, bottom, left))
            
            return filtered, len(filtered) > 0
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return [], False

    def encode_faces(self, image: np.ndarray, 
                     face_locations: List) -> Tuple[List[np.ndarray], bool]:
        """
        Encode detected faces using deep learning.
        
        Args:
            image: Image array in RGB format
            face_locations: List of face locations
            
        Returns:
            Tuple of (list of 128-dim encodings, success flag)
        """
        try:
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

    def _compute_distances(self, face_encoding: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances to all known face encodings.
        
        Args:
            face_encoding: 128-dimensional face encoding
            
        Returns:
            Array of distances to all known encodings
        """
        if len(self._encodings_array) == 0:
            return np.array([])
        
        # Use face_recognition's optimized distance calculation
        return face_recognition.face_distance(self._encodings_array, face_encoding)

    def recognize_face(self, face_encoding: np.ndarray) -> Tuple[Optional[int], float, str]:
        """
        Recognize a single face encoding using k-NN voting or nearest neighbor.
        
        Args:
            face_encoding: 128-dimensional face encoding
            
        Returns:
            Tuple of (person_id, confidence, confidence_level)
            person_id is None if not recognized
            confidence_level is 'high', 'medium', or 'low'
        """
        if len(self.encodings) == 0:
            logger.warning("No encodings loaded for recognition")
            return None, 0.0, "none"
        
        # Compute distances to all known faces
        distances = self._compute_distances(face_encoding)
        
        if USE_KNN_VOTING:
            # K-nearest neighbors voting for more robust recognition
            k = min(KNN_NEIGHBORS, len(distances))
            nearest_indices = np.argsort(distances)[:k]
            nearest_distances = distances[nearest_indices]
            nearest_labels = self.labels[nearest_indices]
            
            # Only consider matches below threshold
            valid_mask = nearest_distances <= DISTANCE_THRESHOLD
            if not np.any(valid_mask):
                return None, float(np.min(distances)), "none"
            
            valid_labels = nearest_labels[valid_mask]
            valid_distances = nearest_distances[valid_mask]
            
            # Vote with distance-weighted confidence
            label_votes = Counter(valid_labels)
            best_label = label_votes.most_common(1)[0][0]
            
            # Calculate confidence based on average distance of winning label
            winning_mask = valid_labels == best_label
            avg_distance = np.mean(valid_distances[winning_mask])
            confidence = 1.0 - min(avg_distance / DISTANCE_THRESHOLD, 1.0)
            
        else:
            # Simple nearest neighbor
            min_distance = np.min(distances)
            min_index = np.argmin(distances)
            
            if min_distance > DISTANCE_THRESHOLD:
                return None, float(min_distance), "none"
            
            best_label = int(self.labels[min_index])
            confidence = 1.0 - min(min_distance / DISTANCE_THRESHOLD, 1.0)
        
        # Determine confidence level
        if confidence >= HIGH_CONFIDENCE:
            level = "high"
        elif confidence >= MIN_CONFIDENCE:
            level = "medium"
        else:
            level = "low"
        
        return int(best_label), float(confidence), level

    def recognize_faces_in_image(self, image: np.ndarray) -> Tuple[List[Dict], bool]:
        """
        Recognize all faces in an image.
        
        Args:
            image: Image array in RGB format
            
        Returns:
            Tuple of (list of recognition results, success flag)
        """
        try:
            # Preprocess
            image = self.preprocess_image(image)
            
            # Detect faces
            face_locations, success = self.detect_faces(image)
            if not success:
                logger.debug("No faces detected in image")
                return [], False
            
            # Encode faces
            face_encodings, success = self.encode_faces(image, face_locations)
            if not success:
                logger.warning("Could not encode detected faces")
                return [], False
            
            # Recognize each face
            results = []
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                person_id, confidence, level = self.recognize_face(encoding)
                
                result = {
                    "location": {
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "left": left
                    },
                    "person_id": person_id,
                    "confidence": confidence,
                    "confidence_level": level,
                    "person_name": None,
                    "image_count": None
                }
                
                if person_id is not None:
                    person_info = self.person_mapping.get(str(person_id), {})
                    result["person_name"] = person_info.get("name")
                    result["image_count"] = person_info.get("image_count")
                
                results.append(result)
            
            recognized = sum(1 for r in results if r["person_id"] is not None)
            logger.info(f"Recognized {recognized}/{len(results)} faces in image")
            return results, True
            
        except Exception as e:
            logger.error(f"Error recognizing faces: {e}")
            return [], False

    def recognize_from_file(self, image_path: str) -> Tuple[List[Dict], Optional[np.ndarray], bool]:
        """
        Recognize faces from an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (recognition results, image array, success flag)
        """
        image, success = self.load_image(image_path)
        if not success:
            return [], None, False
        
        results, success = self.recognize_faces_in_image(image)
        return results, image, success

    def draw_results(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Draw recognition results on image with color-coded confidence.
        
        Args:
            image: Image array in RGB format
            results: List of recognition results
            
        Returns:
            Image with drawn results (BGR format for cv2.imshow)
        """
        # Convert RGB to BGR for cv2
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for result in results:
            top = result["location"]["top"]
            right = result["location"]["right"]
            bottom = result["location"]["bottom"]
            left = result["location"]["left"]
            
            # Color based on confidence level
            level = result.get("confidence_level", "none")
            if level == "high":
                color = (0, 255, 0)      # Green - high confidence
            elif level == "medium":
                color = (0, 255, 255)    # Yellow - medium confidence
            elif level == "low":
                color = (0, 165, 255)    # Orange - low confidence
            else:
                color = (0, 0, 255)      # Red - unknown
            
            # Draw rectangle
            cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 2)
            
            # Draw label with background
            if result["person_name"]:
                label = f"{result['person_name']} ({result['confidence']:.0%})"
            else:
                label = f"Unknown"
            
            # Label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(image_bgr, (left, top - text_height - 10), 
                         (left + text_width + 10, top), color, -1)
            
            # Label text
            cv2.putText(image_bgr, label, (left + 5, top - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image_bgr

    def get_stats(self) -> Dict:
        """Get recognizer statistics"""
        return {
            "total_encodings": len(self.encodings),
            "total_persons": len(self.person_mapping),
            "distance_threshold": DISTANCE_THRESHOLD,
            "detection_model": FACE_DETECTION_MODEL,
            "use_knn": USE_KNN_VOTING,
            "knn_neighbors": KNN_NEIGHBORS
        }
