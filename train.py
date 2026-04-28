"""Training script to encode faces from the dataset"""

import logging
import argparse
import sys
from pathlib import Path

from face_encoder import FaceEncoder
from config import LOGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Train face encodings from dataset"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain even if encodings exist"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Face Recognition Attendance System - Training")
    logger.info("=" * 70)
    
    # Initialize encoder
    encoder = FaceEncoder()
    
    # Train
    if encoder.train():
        # Save encodings
        if encoder.save_encodings():
            stats = encoder.get_stats()
            logger.info(f"\n{'=' * 70}")
            logger.info("Training Summary:")
            logger.info(f"  Total Encodings: {stats['total_encodings']}")
            logger.info(f"  Total Persons: {stats['total_persons']}")
            logger.info(f"  Persons:")
            for person_id, info in stats['person_mapping'].items():
                logger.info(f"    {info['name']}: {info['image_count']} images")
            logger.info(f"{'=' * 70}\n")
            
            logger.info("✓ Training completed successfully!")
            return 0
        else:
            logger.error("✗ Failed to save encodings")
            return 1
    else:
        logger.error("✗ Training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
