"""Setup script for Face Recognition Attendance System"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")

def print_section(text):
    """Print section header"""
    print(f"\n📋 {text}")
    print("-" * 70)

def check_python_version():
    """Check if Python version is compatible"""
    print_section("Checking Python Version")
    
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    print(f"Python version: {version_str}")
    
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    print("✓ Python version is compatible")
    return True

def check_pip():
    """Check if pip is available"""
    print_section("Checking pip")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True
        )
        print(result.stdout.strip())
        print("✓ pip is available")
        return True
    except Exception as e:
        print(f"❌ pip not found: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print_section("Creating Directories")
    
    project_root = Path(__file__).parent
    dirs = ['dataset', 'encodings', 'attendance_records', 'logs']
    
    for dir_name in dirs:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✓ {dir_name}/")
    
    return True

def install_dependencies(upgrade=False):
    """Install Python dependencies"""
    print_section("Installing Dependencies")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"❌ requirements.txt not found: {requirements_file}")
        return False
    
    print("This may take 2-5 minutes on first run...")
    print("Installing packages:\n")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append("-r")
        cmd.append(str(requirements_file))
        
        result = subprocess.run(cmd, text=True)
        
        if result.returncode == 0:
            print("\n✓ All dependencies installed successfully")
            return True
        else:
            print("\n❌ Failed to install some dependencies")
            return False
    except Exception as e:
        print(f"\n❌ Installation error: {e}")
        return False

def verify_imports():
    """Verify that all required modules can be imported"""
    print_section("Verifying Imports")
    
    modules = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("face_recognition", "face_recognition"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - Not installed")
            all_ok = False
    
    return all_ok

def check_dataset():
    """Check dataset structure"""
    print_section("Checking Dataset")
    
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset"
    
    if not dataset_dir.exists():
        print("⚠️  dataset/ directory not found")
        print("Please create dataset/ directory and organize images by person ID")
        return False
    
    person_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    if not person_dirs:
        print("⚠️  No person directories found in dataset/")
        return False
    
    print(f"✓ Found {len(person_dirs)} person directories:")
    
    total_images = 0
    for person_dir in sorted(person_dirs):
        images = [f for f in person_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}]
        print(f"  • {person_dir.name}: {len(images)} images")
        total_images += len(images)
    
    print(f"\n✓ Total images: {total_images}")
    
    if total_images == 0:
        print("⚠️  No images found. Please add face images to dataset/")
        return False
    
    return True

def show_next_steps():
    """Show next steps"""
    print_header("SETUP COMPLETE! ✓")
    
    print("Next Steps:\n")
    print("1. Validate your dataset:")
    print("   > python utils.py validate\n")
    print("2. Train the face recognition model:")
    print("   > python train.py\n")
    print("3. Test the recognition system:")
    print("   > python utils.py test-recognition\n")
    print("4. Start marking attendance:")
    print("   > python attendance.py camera\n")
    print("For detailed help, run:")
    print("   > python quickstart.py\n")

def main():
    """Main setup function"""
    print_header("FACE RECOGNITION ATTENDANCE SYSTEM - SETUP")
    
    # Get user permission
    print("This setup will:")
    print("  1. Check Python version")
    print("  2. Install required dependencies (may take 5-10 minutes)")
    print("  3. Create necessary directories")
    print("  4. Verify all modules are working")
    
    response = input("\nContinue with setup? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\nSetup cancelled.")
        return 1
    
    # Run setup steps
    steps = [
        ("Checking Python version", check_python_version),
        ("Checking pip", check_pip),
        ("Creating directories", create_directories),
        ("Installing dependencies", lambda: install_dependencies(upgrade=False)),
        ("Verifying imports", verify_imports),
        ("Checking dataset", check_dataset),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            return 1
    
    # Show success message
    show_next_steps()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
