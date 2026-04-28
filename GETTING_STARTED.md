# GETTING STARTED GUIDE

## Quick Start (5 Minutes)

### Step 1: Install Dependencies (2 minutes)

Open PowerShell in the project folder and run:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Wait for installation to complete** (this downloads ~200MB of files on first run).

### Step 2: Validate Your Dataset (30 seconds)

```powershell
python utils.py validate
```

You should see output like:
```
Found 3 person directories
  12400794: 11 images
  12400795: 9 images
  12412522: (checking...)

Total persons: 3
Total images: 27+
```

### Step 3: Train the Model (1-2 minutes)

```powershell
python train.py
```

Wait for it to complete. You'll see:
```
Training complete!
Valid persons: 3
Total images processed: 27
Total encodings generated: 27
```

### Step 4: Test Recognition (30 seconds)

```powershell
python utils.py test-recognition
```

Should show:
```
✓ Total encodings: 27
✓ Total persons: 3
✓ Status: Ready for recognition
```

### Step 5: Start Marking Attendance - Choose One:

**Option A: Camera (Real-time, Recommended)**
```powershell
python attendance.py camera
```
- A camera window will open
- Stand in front of camera to be recognized
- Press `q` to quit

**Option B: Test Image**
```powershell
python attendance.py image --image "dataset/12400794/image1.jpg"
```

**Option C: View Summary**
```powershell
python attendance.py summary
```

---

## Full Step-by-Step Guide

### 1. Environment Setup

#### On Windows PowerShell:

```powershell
# Navigate to project
cd "C:\Users\meutk\OneDrive\Desktop\Face Recognition Attendance System"

# Create virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

#### Verify Installation:

```powershell
python -c "import face_recognition; print('✓ face_recognition installed')"
python -c "import cv2; print('✓ opencv installed')"
python -c "import numpy; print('✓ numpy installed')"
```

### 2. Dataset Preparation

Your dataset is already organized. Each folder contains images of one person:

```
dataset/
├── 12400794/          ← Person 1
│   ├── WhatsApp Image 2026-02-05 at 9.02.03 PM.jpeg
│   ├── WhatsApp Image 2026-02-05 at 9.02.04 PM.jpeg
│   └── ... (10 more images)
├── 12400795/          ← Person 2
│   └── ... (9 images)
└── 12412522/          ← Person 3
    └── ... (images)
```

**To add more people:**

1. Create a new folder with person's ID: `dataset/NEW_PERSON_ID/`
2. Add 5-10 photos of that person
3. Retrain: `python train.py`

### 3. Training Phase

```powershell
python train.py
```

**What happens internally:**

1. Loads each image from dataset/
2. Detects faces in each image
3. Creates a 128-dimensional "encoding" (face fingerprint)
4. Saves encodings to `encodings/` folder

**Output files created:**

- `encodings/face_encodings.npy` - All face encodings (27 × 128 matrix)
- `encodings/labels.npy` - Which person is in each encoding (27 labels)
- `encodings/person_mapping.json` - Maps IDs to names

**Training logs:** `logs/training.log`

### 4. Recognition Phase

The system now recognizes faces by:

1. Detecting faces in images/camera
2. Creating an encoding for each new face
3. Comparing to all trained encodings
4. Finding the closest match
5. If confidence > threshold → recognized, else → unknown

### 5. Attendance Marking

#### Option A: Camera Mode (Real-time)

```powershell
python attendance.py camera
```

**Controls:**
- `Q` - Quit
- `S` - Save current frame as image
- `R` - Reset today's attendance

**What you'll see:**
- Green box = recognized person with name and confidence
- Red box = unknown person with distance score

**What happens:**
- First time a person appears → automatically marked present
- Subsequent appearances → logged but not counted again (prevents duplicates)
- Data saved automatically when you quit

#### Option B: Image Mode

```powershell
python attendance.py image --image "C:\path\to\photo.jpg"
```

Or for dataset images:
```powershell
python attendance.py image --image "dataset/12400794/image1.jpg"
```

**Example:**
```powershell
# Recognize a specific image
python attendance.py image --image "dataset/12400795/WhatsApp Image 2026-02-05 at 9.01.21 PM.jpeg"
```

#### Option C: View Results

After marking attendance:

```powershell
python attendance.py summary
```

Shows today's attendance:
```
Date: 2026-02-05
Total Unique Persons: 2
Total Entries: 3

Person Details:
  John Doe (ID: 12400794)
    Entries: 2
    First Entry: 14:30:45
    Last Entry: 14:35:20
    Avg Confidence: 94.23%
```

### 6. Check Attendance Records

Records are saved in CSV files:

```powershell
# View latest attendance
cat "attendance_records/attendance_$(Get-Date -Format 'yyyy-MM-dd').csv"

# List all attendance files
ls "attendance_records/"

# Show attendance using utility
python utils.py show-attendance
```

### 7. Troubleshooting

#### Issue: "ModuleNotFoundError: No module named 'face_recognition'"

```powershell
# Reinstall
pip install --upgrade face-recognition

# If still fails, try:
pip install --force-reinstall face-recognition==1.3.5
```

#### Issue: "No faces detected"

```powershell
# Test detection
python utils.py test-detection

# This shows which images have detectable faces
```

#### Issue: "Camera not working"

```powershell
# Try a different camera ID
python attendance.py camera --camera 1

# Or run without display
python attendance.py camera --no-display
```

#### Issue: Low recognition accuracy

1. **Add more training images** (5-10 per person minimum)
2. **Use varied angles and lighting**
3. **Lower the threshold** in `config.py`:
   ```python
   DISTANCE_THRESHOLD = 0.5  # More strict
   DISTANCE_THRESHOLD = 0.7  # More lenient
   ```

#### Issue: "Out of memory"

In `config.py`, reduce:
```python
NUM_JITTERS = 1  # Was 1, stays 1
FACE_DETECTION_MODEL = "hog"  # Faster
```

---

## Configuration Customization

Edit `config.py` to customize:

```python
# Speed vs Accuracy
FACE_DETECTION_MODEL = "hog"  # "hog" = faster, "cnn" = more accurate
NUM_JITTERS = 1  # Higher = slower but more accurate

# Strictness
DISTANCE_THRESHOLD = 0.6  # Lower = stricter matching (fewer false positives)

# Minimum images per person (must have at least this many)
MIN_IMAGES_PER_PERSON = 3

# Timezone for timestamps
TIMEZONE = "Asia/Kolkata"
```

---

## Advanced Usage

### Batch Process Multiple Images

```powershell
# Edit a batch script or use Python:
python -c "
from attendance import AttendanceMarker
marker = AttendanceMarker()

for i in range(1, 6):
    marker.mark_from_image(f'dataset/12400794/image{i}.jpg')

marker.print_summary()
"
```

### View Model Statistics

```powershell
python utils.py show-encodings
```

### Export Attendance to Excel

```powershell
python advanced_examples.py
# Select option 8: "Export Data for Analysis"
```

---

## File Organization After Basics

After training and running, you'll have:

```
Face Recognition Attendance System/
├── config.py
├── train.py
├── attendance.py
├── README.md
├── ARCHITECTURE.md
│
├── dataset/                          ← Original images
│   ├── 12400794/
│   ├── 12400795/
│   └── 12412522/
│
├── encodings/                        ← Created after training
│   ├── face_encodings.npy           (Face fingerprints)
│   ├── labels.npy                   (Person IDs)
│   └── person_mapping.json          (ID → Name mapping)
│
├── attendance_records/               ← Created after marking
│   ├── attendance_2026-02-05.csv    (Today's attendance)
│   ├── attendance_2026-02-06.csv    (Tomorrow's)
│   └── exports/                     (Exported data)
│
└── logs/                             ← Created on run
    ├── training.log
    ├── face_encoder.log
    ├── face_recognizer.log
    ├── attendance_system.log
    └── attendance.log
```

---

## Performance Tips

1. **For Speed:** Use HOG detection
   ```python
   FACE_DETECTION_MODEL = "hog"
   ```

2. **For Accuracy:** Use CNN detection (requires good GPU)
   ```python
   FACE_DETECTION_MODEL = "cnn"
   ```

3. **Skip Frames:** Process every nth frame
   ```python
   # In attendance.py, increase skip_frames value
   skip_frames = 10  # Process every 10th frame
   ```

4. **Add More Training Data:** More images = better accuracy
   - Collect 10-20 images per person
   - Include different angles and lighting
   - Retrain with `python train.py`

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Train the model with your dataset
3. ✅ Test with camera or images
4. ✅ Check `README.md` for full documentation
5. 📚 Check `ARCHITECTURE.md` to understand how it works
6. 🚀 Check `advanced_examples.py` for advanced features

---

## Interactive Quick Start

Run the interactive guide:

```powershell
python quickstart.py
```

This provides a menu-driven interface with examples and troubleshooting.

---

## Getting Help

1. **Check logs:** Look in `logs/` folder for detailed error messages
2. **Run diagnostics:** `python utils.py validate`
3. **Test components:** `python utils.py test-detection`
4. **Read documentation:** Check `README.md` and `ARCHITECTURE.md`
5. **Check examples:** Run `python advanced_examples.py`

---

Happy Recognition! 👍
