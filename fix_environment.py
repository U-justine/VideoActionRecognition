#!/usr/bin/env python3
"""
Simple environment fix script for Video Action Recognition.
Fixes common numpy and dependency issues.
"""

import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(cmd, description=""):
    """Run a command safely."""
    logging.info(f"Running: {' '.join(cmd)}")
    if description:
        logging.info(f"Purpose: {description}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            logging.info(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error: {e.stderr.strip()}")
        return False

def fix_numpy_issue():
    """Fix numpy version compatibility issues."""
    logging.info("=== Fixing NumPy Compatibility ===")

    # Downgrade numpy to 1.x for compatibility
    success = run_command(
        [sys.executable, '-m', 'pip', 'install', 'numpy<2.0', '--force-reinstall', '--no-cache-dir'],
        "Downgrading NumPy to 1.x for compatibility"
    )

    if success:
        logging.info("✓ NumPy downgrade completed")
    else:
        logging.warning("✗ NumPy downgrade failed")

    return success

def reinstall_core_deps():
    """Reinstall core dependencies."""
    logging.info("=== Reinstalling Core Dependencies ===")

    core_packages = [
        'torch>=2.2.0',
        'torchvision>=0.17.0',
        'transformers==4.43.3',
        'Pillow>=10.0.0',
        'opencv-python>=4.9.0'
    ]

    success_count = 0
    for package in core_packages:
        success = run_command(
            [sys.executable, '-m', 'pip', 'install', package, '--upgrade'],
            f"Installing {package}"
        )
        if success:
            success_count += 1

    logging.info(f"✓ Installed {success_count}/{len(core_packages)} packages")
    return success_count == len(core_packages)

def test_imports():
    """Test if critical imports work."""
    logging.info("=== Testing Imports ===")

    test_modules = [
        ('numpy', 'import numpy as np; print(f"NumPy {np.__version__}")'),
        ('torch', 'import torch; print(f"PyTorch {torch.__version__}")'),
        ('PIL', 'from PIL import Image; print("PIL OK")'),
        ('cv2', 'import cv2; print(f"OpenCV {cv2.__version__}")'),
        ('transformers', 'from transformers import AutoImageProcessor; print("Transformers OK")'),
    ]

    all_good = True
    for name, test_code in test_modules:
        try:
            result = subprocess.run(
                [sys.executable, '-c', test_code],
                capture_output=True, text=True, check=True
            )
            logging.info(f"✓ {name}: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            logging.error(f"✗ {name}: {e.stderr.strip()}")
            all_good = False

    return all_good

def main():
    """Main fix routine."""
    print("🔧 Environment Fix Script")
    print("=" * 40)

    # Step 1: Fix NumPy
    numpy_fixed = fix_numpy_issue()

    # Step 2: Reinstall core dependencies
    deps_fixed = reinstall_core_deps()

    # Step 3: Test everything
    imports_work = test_imports()

    print("\n📊 Results:")
    print(f"   NumPy fixed: {'✓' if numpy_fixed else '✗'}")
    print(f"   Dependencies: {'✓' if deps_fixed else '✗'}")
    print(f"   Imports working: {'✓' if imports_work else '✗'}")

    if imports_work:
        print("\n🎉 Environment fix completed successfully!")
        print("You can now run: streamlit run app.py")
    else:
        print("\n⚠️  Some issues remain. Try:")
        print("1. Recreate virtual environment:")
        print("   rm -rf .venv && python -m venv .venv")
        print("   source .venv/bin/activate")
        print("   pip install -r requirements.txt")
        print("2. Run this script again")

    return 0 if imports_work else 1

if __name__ == "__main__":
    exit(main())
