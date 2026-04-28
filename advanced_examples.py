"""
Advanced Examples - Face Recognition Attendance System
Shows how to extend and customize the system for specific use cases
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from face_encoder import FaceEncoder
from face_recognizer import FaceRecognizer
from attendance_system import AttendanceSystem
from config import ENCODINGS_FILE, LABELS_FILE, PERSON_MAPPING_FILE, ATTENDANCE_DIR

# ============================================================================
# EXAMPLE 1: Custom Training with Image Preprocessing
# ============================================================================

def example_custom_training_with_preprocessing():
    """Advanced training with image preprocessing"""
    print("=" * 70)
    print("EXAMPLE 1: Custom Training with Image Preprocessing")
    print("=" * 70)
    
    # Create custom encoder with preprocessing
    encoder = FaceEncoder()
    
    # Could add preprocessing steps here
    # e.g., brightness adjustment, contrast enhancement, etc.
    
    # Train normally
    encoder.train()
    encoder.save_encodings()
    
    print("✓ Training with preprocessing complete")

# ============================================================================
# EXAMPLE 2: Batch Recognition from Multiple Images
# ============================================================================

def example_batch_recognition():
    """Recognize faces from multiple images at once"""
    print("=" * 70)
    print("EXAMPLE 2: Batch Recognition from Multiple Images")
    print("=" * 70)
    
    import glob
    
    # Load recognizer
    encodings = np.load(ENCODINGS_FILE)
    labels = np.load(LABELS_FILE)
    with open(PERSON_MAPPING_FILE, 'r') as f:
        person_mapping = json.load(f)
    
    recognizer = FaceRecognizer(encodings, labels, person_mapping)
    
    # Get all image files from a folder
    image_folder = "test_images"  # Create this folder with test images
    image_files = glob.glob(f"{image_folder}/*.jpg") + glob.glob(f"{image_folder}/*.jpeg")
    
    results_summary = {}
    
    for image_path in image_files:
        print(f"\nProcessing: {image_path}")
        results, image, success = recognizer.recognize_from_file(image_path)
        
        if success:
            for result in results:
                person_name = result["person_name"] or "Unknown"
                confidence = result["confidence"]
                
                if person_name not in results_summary:
                    results_summary[person_name] = []
                
                results_summary[person_name].append({
                    'image': image_path,
                    'confidence': confidence
                })
                
                print(f"  ✓ {person_name}: {confidence:.2%}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    for person, detections in results_summary.items():
        avg_confidence = np.mean([d['confidence'] for d in detections])
        print(f"  {person}: {len(detections)} detections (avg confidence: {avg_confidence:.2%})")

# ============================================================================
# EXAMPLE 3: Generate Attendance Report
# ============================================================================

def example_attendance_report():
    """Generate detailed attendance report"""
    print("=" * 70)
    print("EXAMPLE 3: Generate Attendance Report")
    print("=" * 70)
    
    # Find all attendance files
    attendance_files = sorted(ATTENDANCE_DIR.glob("attendance_*.csv"))
    
    if not attendance_files:
        print("No attendance records found")
        return
    
    # Combine all records
    all_records = []
    for file in attendance_files:
        df = pd.read_csv(file)
        all_records.append(df)
    
    if not all_records:
        print("No records to process")
        return
    
    combined_df = pd.concat(all_records, ignore_index=True)
    
    # Generate report
    print("\n📊 ATTENDANCE REPORT\n")
    
    print("Total Records:", len(combined_df))
    print("Date Range:", combined_df['timestamp'].min(), "to", combined_df['timestamp'].max())
    
    print("\nAttendance by Person:")
    person_stats = combined_df.groupby('person_name').agg({
        'person_id': 'first',
        'timestamp': 'count',
        'confidence': ['mean', 'min', 'max']
    }).round(4)
    
    print(person_stats)
    
    # Find peak hours
    combined_df['hour'] = pd.to_datetime(combined_df['timestamp']).dt.hour
    print("\nPeak Hours:")
    print(combined_df['hour'].value_counts().sort_index())

# ============================================================================
# EXAMPLE 4: Quality Assessment - Find Low Confidence Detections
# ============================================================================

def example_quality_assessment():
    """Assess recognition quality and find uncertain matches"""
    print("=" * 70)
    print("EXAMPLE 4: Quality Assessment")
    print("=" * 70)
    
    attendance_files = sorted(ATTENDANCE_DIR.glob("attendance_*.csv"))
    
    if not attendance_files:
        print("No attendance records found")
        return
    
    all_records = []
    for file in attendance_files:
        df = pd.read_csv(file)
        all_records.append(df)
    
    if not all_records:
        return
    
    combined_df = pd.concat(all_records, ignore_index=True)
    combined_df['confidence'] = combined_df['confidence'].astype(float)
    
    # Find low confidence detections
    threshold = 0.85
    low_confidence = combined_df[combined_df['confidence'] < threshold]
    
    print(f"\nDetections with confidence < {threshold}:")
    print(f"Count: {len(low_confidence)}")
    
    if len(low_confidence) > 0:
        print("\nDetails:")
        for idx, row in low_confidence.iterrows():
            print(f"  {row['person_name']}: {row['confidence']:.4f} at {row['timestamp']}")
    
    # Statistics
    print("\nConfidence Statistics:")
    print(f"  Mean: {combined_df['confidence'].mean():.4f}")
    print(f"  Min: {combined_df['confidence'].min():.4f}")
    print(f"  Max: {combined_df['confidence'].max():.4f}")
    print(f"  Std: {combined_df['confidence'].std():.4f}")

# ============================================================================
# EXAMPLE 5: Attendance Analytics
# ============================================================================

def example_attendance_analytics():
    """Perform advanced attendance analytics"""
    print("=" * 70)
    print("EXAMPLE 5: Attendance Analytics")
    print("=" * 70)
    
    attendance_files = sorted(ATTENDANCE_DIR.glob("attendance_*.csv"))
    
    if not attendance_files:
        print("No attendance records found")
        return
    
    # Process each day
    print("\nDaily Summary:\n")
    
    for file in attendance_files:
        df = pd.read_csv(file)
        date = file.stem.replace('attendance_', '')
        
        unique_persons = df['person_name'].nunique()
        total_entries = len(df)
        
        print(f"Date: {date}")
        print(f"  Unique Persons: {unique_persons}")
        print(f"  Total Entries: {total_entries}")
        
        # Early arrivals
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('timestamp')
        
        print(f"  Arrival Times:")
        first_entries = df_sorted.drop_duplicates('person_name', keep='first')
        for _, row in first_entries.iterrows():
            print(f"    {row['person_name']}: {row['timestamp'].strftime('%H:%M:%S')}")
        print()

# ============================================================================
# EXAMPLE 6: Export Encodings Analysis
# ============================================================================

def example_encodings_analysis():
    """Analyze the trained encodings"""
    print("=" * 70)
    print("EXAMPLE 6: Encodings Analysis")
    print("=" * 70)
    
    # Load encodings
    encodings = np.load(ENCODINGS_FILE)
    labels = np.load(LABELS_FILE)
    
    with open(PERSON_MAPPING_FILE, 'r') as f:
        person_mapping = json.load(f)
    
    print("\nEncoding Statistics:\n")
    print(f"Shape: {encodings.shape}")
    print(f"Data Type: {encodings.dtype}")
    
    # Per-person statistics
    print("\nPer-Person Encoding Statistics:")
    
    for person_id, person_info in person_mapping.items():
        person_id_int = int(person_id)
        person_encodings = encodings[labels == person_id_int]
        
        print(f"\n{person_info['name']}:")
        print(f"  Encodings: {len(person_encodings)}")
        print(f"  Mean: {np.mean(person_encodings):.6f}")
        print(f"  Std: {np.std(person_encodings):.6f}")
        print(f"  Min: {np.min(person_encodings):.6f}")
        print(f"  Max: {np.max(person_encodings):.6f}")

# ============================================================================
# EXAMPLE 7: Custom Distance Threshold Analysis
# ============================================================================

def example_threshold_analysis():
    """Analyze how different thresholds affect recognition"""
    print("=" * 70)
    print("EXAMPLE 7: Distance Threshold Analysis")
    print("=" * 70)
    
    # Load encodings
    encodings = np.load(ENCODINGS_FILE)
    labels = np.load(LABELS_FILE)
    
    import face_recognition
    
    # Calculate all pairwise distances within same person
    with open(PERSON_MAPPING_FILE, 'r') as f:
        person_mapping = json.load(f)
    
    print("\nIntra-person distance statistics:\n")
    
    for person_id, person_info in person_mapping.items():
        person_id_int = int(person_id)
        person_encodings = encodings[labels == person_id_int]
        
        if len(person_encodings) < 2:
            continue
        
        # Calculate distances between encodings of same person
        distances = []
        for i in range(len(person_encodings)):
            for j in range(i+1, len(person_encodings)):
                dist = face_recognition.face_distance(
                    [person_encodings[i]], 
                    person_encodings[j]
                )[0]
                distances.append(dist)
        
        if distances:
            print(f"{person_info['name']}:")
            print(f"  Min distance: {np.min(distances):.4f}")
            print(f"  Max distance: {np.max(distances):.4f}")
            print(f"  Mean distance: {np.mean(distances):.4f}")
            print(f"  Std distance: {np.std(distances):.4f}")
            print()

# ============================================================================
# EXAMPLE 8: Export Data for External Analysis
# ============================================================================

def example_export_for_analysis():
    """Export data in various formats for external analysis"""
    print("=" * 70)
    print("EXAMPLE 8: Export Data for External Analysis")
    print("=" * 70)
    
    attendance_files = sorted(ATTENDANCE_DIR.glob("attendance_*.csv"))
    
    if not attendance_files:
        print("No attendance records found")
        return
    
    # Combine all records
    all_records = []
    for file in attendance_files:
        df = pd.read_csv(file)
        all_records.append(df)
    
    if not all_records:
        return
    
    combined_df = pd.concat(all_records, ignore_index=True)
    
    # Export to different formats
    output_dir = ATTENDANCE_DIR / "exports"
    output_dir.mkdir(exist_ok=True)
    
    # CSV
    csv_file = output_dir / "attendance_export.csv"
    combined_df.to_csv(csv_file, index=False)
    print(f"✓ Exported to CSV: {csv_file}")
    
    # JSON
    json_file = output_dir / "attendance_export.json"
    combined_df.to_json(json_file, orient='records', date_format='iso')
    print(f"✓ Exported to JSON: {json_file}")
    
    # Excel (if openpyxl is available)
    try:
        excel_file = output_dir / "attendance_export.xlsx"
        combined_df.to_excel(excel_file, index=False)
        print(f"✓ Exported to Excel: {excel_file}")
    except ImportError:
        print("⚠️  openpyxl not installed, skipping Excel export")

# ============================================================================
# Main Menu
# ============================================================================

def main():
    """Run examples"""
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ADVANCED EXAMPLES - Face Recognition" + " " * 17 + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    examples = [
        ("Custom Training with Preprocessing", example_custom_training_with_preprocessing),
        ("Batch Recognition from Multiple Images", example_batch_recognition),
        ("Generate Attendance Report", example_attendance_report),
        ("Quality Assessment", example_quality_assessment),
        ("Attendance Analytics", example_attendance_analytics),
        ("Encodings Analysis", example_encodings_analysis),
        ("Distance Threshold Analysis", example_threshold_analysis),
        ("Export Data for Analysis", example_export_for_analysis),
    ]
    
    while True:
        print("\nSelect an example to run:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"  {i}. {name}")
        print(f"  {len(examples) + 1}. Exit")
        
        choice = input("\nEnter choice (1-{}): ".format(len(examples) + 1)).strip()
        
        try:
            choice_idx = int(choice) - 1
            
            if choice_idx == len(examples):
                print("\nGoodbye! 👋\n")
                break
            elif 0 <= choice_idx < len(examples):
                print()
                examples[choice_idx][1]()
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid input")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("⚠️  This script requires pandas. Install with: pip install pandas")
        print("Some examples may not work.\n")
    
    main()
