"""Quick start guide and examples"""

import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_header(text):
    """Print formatted header"""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")

def show_quick_start():
    """Show quick start guide"""
    print_header("QUICK START GUIDE - Face Recognition Attendance System")
    
    print("📋 STEP 1: Install Dependencies")
    print("-" * 70)
    print("Run the following command to install all required packages:")
    print("\n  pip install -r requirements.txt\n")
    print("⏱️  This will take 2-5 minutes on first run.\n")
    
    print("📋 STEP 2: Validate Dataset")
    print("-" * 70)
    print("Verify your face dataset is properly organized:")
    print("\n  python utils.py validate\n")
    print("Expected output:")
    print("  ✓ Found 3 person directories")
    print("  ✓ Total images: 30+\n")
    
    print("📋 STEP 3: Train the Model")
    print("-" * 70)
    print("Generate face encodings from your dataset:")
    print("\n  python train.py\n")
    print("Expected output:")
    print("  ✓ Processing person: 12400794")
    print("  ✓ Training complete!")
    print("  ✓ Total encodings: 30\n")
    
    print("📋 STEP 4: Test Recognition")
    print("-" * 70)
    print("Verify the trained model works:")
    print("\n  python utils.py test-recognition\n")
    print("Expected output:")
    print("  ✓ Total encodings: 30")
    print("  ✓ Total persons: 3")
    print("  ✓ Status: Ready for recognition\n")
    
    print("📋 STEP 5: Mark Attendance")
    print("-" * 70)
    print("Choose your method:\n")
    print("  Option A: From Camera (Real-time)")
    print("    python attendance.py camera")
    print("    Controls: q=quit, s=save frame, r=reset\n")
    print("  Option B: From Image File")
    print("    python attendance.py image --image path/to/photo.jpg\n")
    print("  Option C: View Attendance Summary")
    print("    python attendance.py summary\n")

def show_examples():
    """Show usage examples"""
    print_header("USAGE EXAMPLES")
    
    print("1️⃣  TRAINING")
    print("-" * 70)
    print("  python train.py")
    print("  → Trains encodings from dataset/\n")
    
    print("2️⃣  CAMERA-BASED ATTENDANCE (Real-time)")
    print("-" * 70)
    print("  python attendance.py camera")
    print("  → Starts webcam, marks attendance for recognized people")
    print("  → Press 'q' to quit\n")
    
    print("3️⃣  IMAGE-BASED ATTENDANCE")
    print("-" * 70)
    print("  python attendance.py image --image group_photo.jpg")
    print("  → Recognizes faces in image and marks attendance\n")
    
    print("4️⃣  ATTENDANCE SUMMARY")
    print("-" * 70)
    print("  python attendance.py summary")
    print("  → Shows today's attendance statistics\n")
    
    print("5️⃣  DATASET VALIDATION")
    print("-" * 70)
    print("  python utils.py validate")
    print("  → Checks dataset structure and image counts\n")
    
    print("6️⃣  TEST FACE DETECTION")
    print("-" * 70)
    print("  python utils.py test-detection")
    print("  → Tests if faces are being detected in sample images\n")
    
    print("7️⃣  SHOW TRAINING INFO")
    print("-" * 70)
    print("  python utils.py show-encodings")
    print("  → displays trained encodings and person info\n")
    
    print("8️⃣  VIEW ATTENDANCE RECORDS")
    print("-" * 70)
    print("  python utils.py show-attendance")
    print("  → Shows all attendance records saved\n")

def show_troubleshooting():
    """Show troubleshooting guide"""
    print_header("TROUBLESHOOTING GUIDE")
    
    print("❌ Issue: 'ModuleNotFoundError: No module named face_recognition'")
    print("-" * 70)
    print("Solution: Run -> pip install -r requirements.txt\n")
    
    print("❌ Issue: No faces detected in images")
    print("-" * 70)
    print("Solutions:")
    print("  1. Check image quality and lighting")
    print("  2. Ensure faces are clear and at least 20x20 pixels")
    print("  3. Run: python utils.py test-detection\n")
    
    print("❌ Issue: Low recognition confidence (<80%)")
    print("-" * 70)
    print("Solutions:")
    print("  1. Add more training images per person (5-10 minimum)")
    print("  2. Use varied angles and lighting in training data")
    print("  3. Lower DISTANCE_THRESHOLD in config.py\n")
    
    print("❌ Issue: Camera not working")
    print("-" * 70)
    print("Solutions:")
    print("  1. Check camera is connected and recognized by OS")
    print("  2. Try: python attendance.py camera --camera 1")
    print("  3. Try: python attendance.py camera --no-display\n")
    
    print("❌ Issue: Out of memory error")
    print("-" * 70)
    print("Solutions:")
    print("  1. Reduce NUM_JITTERS in config.py (1 or 2)")
    print("  2. Use HOG instead of CNN in config.py")
    print("  3. Process smaller/fewer images\n")

def show_system_info():
    """Show system information"""
    print_header("SYSTEM INFORMATION")
    
    print("Python Version:", sys.version.split()[0])
    print("Operating System:", sys.platform)
    print("Project Directory:", project_root)
    
    print("\nDirectory Structure:")
    dirs_to_check = ['dataset', 'encodings', 'attendance_records', 'logs']
    for dir_name in dirs_to_check:
        dir_path = project_root / dir_name
        status = "✓" if dir_path.exists() else "✗"
        print(f"  {status} {dir_name}/")
    
    print("\nKey Files:")
    files_to_check = ['train.py', 'attendance.py', 'config.py', 'requirements.txt']
    for file_name in files_to_check:
        file_path = project_root / file_name
        status = "✓" if file_path.exists() else "✗"
        print(f"  {status} {file_name}")

def main():
    """Main menu"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "FACE RECOGNITION ATTENDANCE SYSTEM" + " " * 24 + "║")
    print("║" + " " * 13 + "Quick Start & Usage Guide" + " " * 31 + "║")
    print("╚" + "═" * 68 + "╝")
    
    while True:
        print("\nSelect an option:")
        print("  1. Show Quick Start Guide")
        print("  2. Show Usage Examples")
        print("  3. Show Troubleshooting Guide")
        print("  4. Show System Information")
        print("  5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            show_quick_start()
        elif choice == "2":
            show_examples()
        elif choice == "3":
            show_troubleshooting()
        elif choice == "4":
            show_system_info()
        elif choice == "5":
            print("\nGoodbye! 👋\n")
            break
        else:
            print("\n❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting... 👋\n")
        sys.exit(0)
