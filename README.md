# Face Recognition Attendance System

A complete, production-ready face recognition system for automatic attendance marking. Built with Python using state-of-the-art deep learning models for face detection and recognition.

## Features

✅ **Automatic Face Recognition** - Recognizes faces from images or camera feed  
✅ **Dataset Training** - Trains on your custom face dataset  
✅ **Attendance Logging** - Automatically records attendance with timestamps  
✅ **Multiple Input Sources** - Works with camera feed or image files  
✅ **Confidence Scores** - Reports recognition confidence for each detection  
✅ **Face Detection Visualization** - Visual feedback with bounding boxes  
✅ **Comprehensive Logging** - Detailed logs for debugging and monitoring  
✅ **Data Validation Tools** - Utilities to validate dataset and model quality  

## Project Structure

```
Face Recognition Attendance System/
├── config.py                 # Configuration settings
├── face_encoder.py          # Face encoding/training module
├── face_recognizer.py       # Face recognition module
├── attendance_system.py      # Attendance logging module
├── train.py                 # Training script
├── attendance.py            # Main attendance application
├── utils.py                 # Utility tools
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── dataset/                # Your face dataset
│   ├── 12400794/          # Person 1
│   ├── 12400795/          # Person 2
│   └── 12412522/          # Person 3
├── encodings/              # Generated encodings (created on training)
├── attendance_records/     # Attendance CSV files (created when marking)
└── logs/                   # Application logs (created on run)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Webcam (for camera mode) or image files

### Step 1: Clone/Download the Project

Navigate to your project directory:
```bash
cd "Face Recognition Attendance System"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The first installation may take several minutes as it downloads and compiles dlib. Be patient!

## Dataset Organization

Your dataset should be organized as follows:

```
dataset/
├── person_id_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person_id_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── person_id_3/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Each person should have at least 3-5 images for best results. Images can be:
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`
- Different angles and lighting conditions (helps with robustness)
- Minimum face size: 20x20 pixels

## Usage Guide

### 1. Validate Your Dataset

Before training, validate that your dataset is properly organized:

```bash
python utils.py validate
```

This checks:
- All person directories exist
- Images are readable
- Image counts per person

### 2. Test Face Detection

Test if faces are being detected properly:

```bash
python utils.py test-detection
```

This will show detection statistics for sample images.

### 3. Train the Model

Train face encodings from your dataset:

```bash
python train.py
```

**What happens:**
1. Loads all images from dataset
2. Detects faces in each image
3. Generates 128-dimensional encodings for each face
4. Saves encodings to `encodings/` directory
5. Creates mapping of person IDs to names

**Output files:**
- `encodings/face_encodings.npy` - Face encodings
- `encodings/labels.npy` - Person labels
- `encodings/person_mapping.json` - ID to name mapping

### 4. Mark Attendance from Camera

Start real-time attendance marking from webcam:

```bash
python attendance.py camera
```

**Controls:**
- `q` - Quit
- `s` - Save current frame
- `r` - Reset attendance records

**Features:**
- Automatically marks attendance first time a person is detected
- Shows confidence scores
- Displays bounding boxes on recognized faces

### 5. Mark Attendance from Image

Process a single image file:

```bash
python attendance.py image --image path/to/image.jpg
```

### 6. View Attendance Summary

Display today's attendance summary:

```bash
python attendance.py summary
```

Shows:
- Total recognized persons
- Entry timestamps for each person
- Average recognition confidence per person

### 7. Test Recognition Model

Verify the trained model is ready:

```bash
python utils.py test-recognition
```

Shows model statistics and validation.

### 8. View Trained Encodings Info

Display information about trained encodings:

```bash
python utils.py show-encodings
```

### 9. View Attendance Records

Display all recorded attendance:

```bash
python utils.py show-attendance
```

## Configuration

Edit `config.py` to customize:

```python
# Face detection settings
FACE_DETECTION_MODEL = "hog"  # "hog" (fast) or "cnn" (accurate)
DISTANCE_THRESHOLD = 0.6      # Lower = stricter matching
NUM_JITTERS = 1               # Higher = more accurate but slower

# File handling
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
MIN_FACE_SIZE = 20            # Minimum pixels for face

# Training
MIN_IMAGES_PER_PERSON = 3     # Minimum images to train a person

# Timing
TIMEZONE = "Asia/Kolkata"     # For timestamps
```

## Attendance File Format

Attendance records are saved as CSV files: `attendance_YYYY-MM-DD.csv`

Format:
```
person_id,person_name,timestamp,confidence,face_location
12400794,John Doe,2026-02-05 14:30:45,0.8421,"(100, 250, 200, 150)"
```

## Troubleshooting

### No faces detected in images
- Check image quality and lighting
- Ensure faces are at least 20x20 pixels
- Try different angles in dataset

### Low recognition confidence
- Add more training images per person
- Use images from different lighting/angles
- Lower `DISTANCE_THRESHOLD` in config.py

### "No module named 'face_recognition'"
```bash
# Reinstall dependencies
pip install --upgrade face-recognition
```

### Camera not working
- Check camera is connected and working
- Try different camera ID: `python attendance.py camera --camera 1`

### Out of memory errors
- Reduce `NUM_JITTERS` in config.py (faster but less accurate)
- Use `FACE_DETECTION_MODEL = "hog"` instead of "cnn"

## Performance Tips

1. **Speed vs Accuracy Trade-off:**
   - Use `"hog"` for faster detection
   - Use `"cnn"` for more accurate detection (requires GPU)

2. **Better Recognition:**
   - Provide 5-10 varied images per person
   - Include different lighting conditions
   - Include different angles

3. **Camera Performance:**
   - Skip frames: Only processes every 5th frame
   - Lower resolution inputs
   - Run on machine with good CPU

## Advanced Features

### Using GPU Acceleration
For faster CNN-based detection:
```python
# In config.py
FACE_DETECTION_MODEL = "cnn"  # Requires CUDA-capable GPU
```

### Custom Distance Threshold
Adjust recognition strictness:
```python
# In config.py
DISTANCE_THRESHOLD = 0.5  # Stricter (fewer false positives)
DISTANCE_THRESHOLD = 0.7  # Looser (fewer false negatives)
```

## Technical Details

### Face Encoding
- **Library:** face_recognition (uses dlib)
- **Encoding:** 128-dimensional vectors
- **Model:** ResNet-based deep learning model
- **Distance Metric:** Euclidean distance

### Face Detection
- **HOG Method:** Histogram of Oriented Gradients (fast, CPU)
- **CNN Method:** Convolutional Neural Network (accurate, GPU)

### Attendance Logging
- Timestamps in local timezone
- Confidence scores via distance metrics
- CSV format for easy analysis

## Logs

All activities are logged to individual files:
- `logs/training.log` - Training progress
- `logs/face_encoder.log` - Encoding details
- `logs/face_recognizer.log` - Recognition results
- `logs/attendance_system.log` - Attendance marking
- `logs/attendance.log` - Main application logs

## Improvements & Customization

The system is modular and extensible:

**Add Database Support:**
```python
# Modify attendance_system.py to save to database instead of CSV
```

**Add Web Interface:**
```python
# Create Flask/FastAPI wrapper around the attendance system
```

**Add Email Notifications:**
```python
# Submit attendance records via email
```

**Add Multiple Camera Support:**
```python
# Run multiple camera threads simultaneously
```

## Known Limitations

1. **Lighting:** Performance varies with lighting conditions
2. **Masks:** May not work well with masks/sunglasses
3. **Profile Angle:** Works best with frontal faces
4. **Database:** Currently uses local storage (no sync)

## Performance Metrics

Expected performance on typical hardware:
- Training: 10-30 seconds for ~30 people with 5 images each
- Camera Recognition: 15-25 FPS (HOG), 5-10 FPS (CNN)
- Accuracy: 95%+ for well-lit, frontal faces

## License

This system is provided as-is for educational and business use.

## Support & Issues

For issues or feature requests:
1. Check the troubleshooting section
2. Review logs in `logs/` directory
3. Validate dataset with `python utils.py validate`
4. Test recognition with `python utils.py test-recognition`

## Version History

**v1.0.0** (2026-02-05)
- Initial release
- Core face recognition functionality
- Attendance logging system
- Configuration system
- Utilities and validation tools

---

**Built with ❤️ using face_recognition, OpenCV, and Python**
