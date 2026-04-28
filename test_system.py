"""
Test suite for Face Recognition Attendance System
Validates installation and system components
"""

import sys
import subprocess
from pathlib import Path
import importlib

def print_header(text, char="="):
    """Print formatted header"""
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")

def test_python_version():
    """Test Python version compatibility"""
    print("Testing Python version...", end=" ")
    version_info = sys.version_info
    
    if version_info.major >= 3 and version_info.minor >= 8:
        print(f"✓ ({version_info.major}.{version_info.minor}.{version_info.micro})")
        return True
    else:
        print(f"✗ (requires 3.8+, have {version_info.major}.{version_info.minor})")
        return False

def test_module_import(module_name, display_name):
    """Test if a module can be imported"""
    print(f"Testing {display_name}...", end=" ")
    try:
        importlib.import_module(module_name)
        print("✓")
        return True
    except ImportError as e:
        print(f"✗ ({e})")
        return False

def test_required_modules():
    """Test all required modules"""
    modules = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("face_recognition", "face_recognition"),
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"),
        ("pandas", "Pandas"),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        if not test_module_import(module_name, display_name):
            all_ok = False
    
    return all_ok

def test_project_structure():
    """Test project directory structure"""
    print("\nTesting project structure...")
    
    project_root = Path(__file__).parent
    required_files = [
        'config.py',
        'face_encoder.py',
        'face_recognizer.py',
        'attendance_system.py',
        'train.py',
        'attendance.py',
        'utils.py',
        'requirements.txt',
        'README.md',
    ]
    
    all_ok = True
    for file_name in required_files:
        file_path = project_root / file_name
        status = "✓" if file_path.exists() else "✗"
        print(f"  {status} {file_name}")
        if not file_path.exists():
            all_ok = False
    
    return all_ok

def test_dataset_structure():
    """Test dataset organization"""
    print("\nTesting dataset structure...")
    
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset"
    
    if not dataset_dir.exists():
        print("  ✗ dataset/ directory not found")
        return False
    
    person_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    if not person_dirs:
        print("  ✗ No person directories found in dataset/")
        return False
    
    print(f"  ✓ Found {len(person_dirs)} person directories:")
    
    all_ok = True
    total_images = 0
    
    for person_dir in sorted(person_dirs):
        images = [f for f in person_dir.iterdir() 
                 if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}]
        
        if images:
            print(f"    • {person_dir.name}: {len(images)} images")
            total_images += len(images)
        else:
            print(f"    ✗ {person_dir.name}: No images")
            all_ok = False
    
    if total_images > 0:
        print(f"  ✓ Total images: {total_images}")
    else:
        print("  ✗ No images found in dataset")
        all_ok = False
    
    return all_ok

def test_local_imports():
    """Test importing project modules"""
    print("\nTesting project module imports...")
    
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    modules_to_test = [
        ('config', 'Configuration'),
        ('face_encoder', 'Face Encoder'),
        ('face_recognizer', 'Face Recognizer'),
        ('attendance_system', 'Attendance System'),
    ]
    
    all_ok = True
    for module_name, display_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {display_name}")
        except Exception as e:
            print(f"  ✗ {display_name}: {e}")
            all_ok = False
    
    return all_ok

def test_camera_access():
    """Test if camera is accessible"""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            cap.release()
            print("  ✓ Camera is accessible")
            return True
        else:
            print("  ⚠️  Camera not found or not accessible")
            print("    (This is ok if you don't have a camera)")
            return True  # Not a failure
    except Exception as e:
        print(f"  ⚠️  Could not test camera: {e}")
        return True

def test_face_detection():
    """Test if face detection is working"""
    print("\nTesting face detection...")
    
    try:
        import face_recognition
        import numpy as np
        
        # Create a dummy image
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Try to detect faces
        faces = face_recognition.face_locations(dummy_image, model='hog')
        print("  ✓ Face detection is working")
        return True
    except Exception as e:
        print(f"  ✗ Face detection error: {e}")
        return False

def test_encoding_operations():
    """Test face encoding operations"""
    print("\nTesting face encoding operations...")
    
    try:
        import face_recognition
        import numpy as np
        
        # Create a dummy face location
        dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        face_location = (20, 180, 220, 20)
        
        # Try to encode
        encodings = face_recognition.face_encodings(dummy_image, [face_location])
        
        if len(encodings) > 0:
            print("  ✓ Face encoding is working")
            return True
        else:
            print("  ⚠️  No encoding generated (face may be too small)")
            return True
    except Exception as e:
        print(f"  ✗ Face encoding error: {e}")
        return False

def test_directories_writable():
    """Test if required directories are writable"""
    print("\nTesting directory write access...")
    
    project_root = Path(__file__).parent
    dirs_to_check = {
        'encodings': 'Encodings',
        'attendance_records': 'Attendance Records',
        'logs': 'Logs',
    }
    
    all_ok = True
    for dir_name, display_name in dirs_to_check.items():
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        
        try:
            test_file = dir_path / ".write_test"
            test_file.touch()
            test_file.unlink()
            print(f"  ✓ {display_name} directory is writable")
        except Exception as e:
            print(f"  ✗ {display_name} directory not writable: {e}")
            all_ok = False
    
    return all_ok

def main():
    """Run all tests"""
    print_header("FACE RECOGNITION ATTENDANCE SYSTEM - TEST SUITE")
    
    tests = [
        ("Python Version", test_python_version),
        ("Required Modules", test_required_modules),
        ("Project Files", test_project_structure),
        ("Dataset", test_dataset_structure),
        ("Project Imports", test_local_imports),
        ("Directory Access", test_directories_writable),
        ("Face Detection", test_face_detection),
        ("Face Encoding", test_encoding_operations),
        ("Camera", test_camera_access),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python train.py")
        print("  2. Run: python attendance.py camera")
        print("  3. Run: python quickstart.py (for interactive guide)")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        print("\nPlease fix the issues above and run this test again.")
        print("\nFor help:")
        print("  • Check GETTING_STARTED.md")
        print("  • Check README.md")
        print("  • Run: python quickstart.py")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
