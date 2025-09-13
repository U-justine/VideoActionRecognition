#!/usr/bin/env python3
"""
Script to diagnose and fix the numpy availability issue in video action recognition.
This script will check the current environment and attempt to fix common issues.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    if description:
        print(f"Purpose: {description}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Success: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e.stderr.strip()}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def check_virtual_env():
    """Check if we're in a virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    venv_path = os.environ.get('VIRTUAL_ENV')

    print("=== Virtual Environment Status ===")
    print(f"In virtual environment: {in_venv}")
    print(f"Virtual env path: {venv_path}")
    print(f"Python executable: {sys.executable}")
    print()

    return in_venv

def test_numpy_import():
    """Test if numpy can be imported and used."""
    print("=== Testing Numpy Import ===")

    try:
        import numpy as np
        print(f"✓ Numpy imported successfully")
        print(f"✓ Numpy version: {np.__version__}")

        # Test basic operations
        arr = np.array([1, 2, 3])
        result = arr * 2
        print(f"✓ Basic operations work: {result}")

        # Test the specific operations used in video processing
        test_array = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        stacked = np.stack([test_array, test_array], axis=0)
        print(f"✓ Stack operations work, shape: {stacked.shape}")

        return True

    except ImportError as e:
        print(f"✗ Cannot import numpy: {e}")
        return False
    except Exception as e:
        print(f"✗ Numpy operations failed: {e}")
        return False

def test_dependencies():
    """Test all required dependencies."""
    print("=== Testing Dependencies ===")

    dependencies = [
        ('numpy', 'import numpy; print(numpy.__version__)'),
        ('torch', 'import torch; print(torch.__version__)'),
        ('PIL', 'from PIL import Image; print("PIL OK")'),
        ('cv2', 'import cv2; print(cv2.__version__)'),
        ('transformers', 'import transformers; print(transformers.__version__)'),
    ]

    all_ok = True
    for name, test_cmd in dependencies:
        try:
            result = subprocess.run([sys.executable, '-c', test_cmd],
                                  capture_output=True, text=True, check=True)
            print(f"✓ {name}: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"✗ {name}: {e.stderr.strip()}")
            all_ok = False
        except Exception as e:
            print(f"✗ {name}: {e}")
            all_ok = False

    print()
    return all_ok

def fix_numpy_installation():
    """Attempt to fix numpy installation issues."""
    print("=== Fixing Numpy Installation ===")

    fixes = [
        # Upgrade pip first
        ([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
         "Upgrading pip"),

        # Force reinstall numpy
        ([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-cache-dir', 'numpy>=1.24.0'],
         "Force reinstalling numpy"),

        # Install other required packages
        ([sys.executable, '-m', 'pip', 'install', '--upgrade', 'Pillow>=10.0.0'],
         "Upgrading Pillow"),

        ([sys.executable, '-m', 'pip', 'install', '--upgrade', 'opencv-python>=4.9.0'],
         "Upgrading OpenCV"),

        # Install from requirements.txt
        ([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
         "Installing from requirements.txt"),
    ]

    for cmd, desc in fixes:
        success = run_command(cmd, desc)
        if not success:
            print(f"Warning: {desc} failed, continuing...")
        print()

def create_activation_script():
    """Create a script to properly activate the virtual environment."""
    script_content = '''#!/bin/bash
# Script to activate virtual environment and run the app

# Get the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$DIR/.venv/bin/activate"

# Check if activation worked
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment activated: $VIRTUAL_ENV"

    # Verify numpy is available
    python -c "import numpy; print(f'✓ Numpy version: {numpy.__version__}')" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ Numpy is available"
    else
        echo "✗ Numpy still not available, running fix script..."
        python fix_numpy_issue.py
    fi

    # Run the app
    echo "Starting Streamlit app..."
    streamlit run app.py
else
    echo "✗ Failed to activate virtual environment"
    echo "Try running: source .venv/bin/activate"
fi
'''

    with open('run_app.sh', 'w') as f:
        f.write(script_content)

    # Make executable
    os.chmod('run_app.sh', 0o755)
    print("✓ Created run_app.sh script")

def main():
    """Main diagnostic and fix routine."""
    print("Video Action Recognition - Numpy Fix Script")
    print("=" * 50)

    # Check virtual environment
    in_venv = check_virtual_env()

    if not in_venv:
        print("⚠️  Warning: Not in virtual environment!")
        print("Please activate your virtual environment first:")
        print("source .venv/bin/activate")
        print()

    # Test current state
    numpy_ok = test_numpy_import()
    deps_ok = test_dependencies()

    if numpy_ok and deps_ok:
        print("✅ All dependencies are working correctly!")
        print("The numpy issue might be intermittent or environment-specific.")
        print("Try running the app again.")
    else:
        print("🔧 Attempting to fix issues...")
        fix_numpy_installation()

        print("=== Re-testing after fixes ===")
        numpy_ok = test_numpy_import()

        if numpy_ok:
            print("✅ Numpy issue fixed!")
        else:
            print("❌ Numpy issue persists. Additional steps needed:")
            print("1. Try recreating the virtual environment:")
            print("   rm -rf .venv")
            print("   python -m venv .venv")
            print("   source .venv/bin/activate")
            print("   pip install -r requirements.txt")
            print()
            print("2. Check for system-level conflicts")
            print("3. Try a different Python version")

    # Create helper script
    create_activation_script()

    print("\n=== Next Steps ===")
    print("1. Make sure virtual environment is activated:")
    print("   source .venv/bin/activate")
    print("2. Or use the helper script:")
    print("   ./run_app.sh")
    print("3. Then run your app:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()
