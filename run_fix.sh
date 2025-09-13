#!/bin/bash

# Script to fix numpy availability issue in Video Action Recognition
# This script handles the directory with spaces in the name

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Video Action Recognition - Numpy Fix Script"
echo "============================================"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Check if we're in the right directory
if [[ ! -f "$SCRIPT_DIR/requirements.txt" ]]; then
    echo "❌ Error: requirements.txt not found"
    echo "Make sure you're running this script from the Video Action Recognition directory"
    exit 1
fi

# Check if virtual environment exists
if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
    echo "❌ Error: Virtual environment not found"
    echo "Creating virtual environment..."
    cd "$SCRIPT_DIR"
    python3 -m venv .venv
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$SCRIPT_DIR/.venv/bin/activate"

if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip first
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Check current numpy status
echo ""
echo "Checking current numpy status..."
python -c "import numpy; print(f'✅ Numpy version: {numpy.__version__}')" 2>/dev/null
NUMPY_STATUS=$?

if [[ $NUMPY_STATUS -eq 0 ]]; then
    echo "✅ Numpy is already working"
else
    echo "❌ Numpy not available, fixing..."

    # Force reinstall numpy
    echo "Force reinstalling numpy..."
    python -m pip install --force-reinstall --no-cache-dir "numpy>=1.24.0"

    # Install other dependencies
    echo "Installing/updating other dependencies..."
    python -m pip install --upgrade "Pillow>=10.0.0"
    python -m pip install --upgrade "opencv-python>=4.9.0"

    # Install all requirements
    echo "Installing from requirements.txt..."
    python -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# Final test
echo ""
echo "Testing final configuration..."
python -c "
try:
    import numpy as np
    print(f'✅ Numpy: {np.__version__}')

    import torch
    print(f'✅ PyTorch: {torch.__version__}')

    from PIL import Image
    print('✅ PIL: Available')

    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')

    from transformers import AutoImageProcessor
    print('✅ Transformers: Available')

    # Test the specific numpy operations used in video processing
    test_array = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
    stacked = np.stack([test_array, test_array], axis=0)
    print(f'✅ Numpy operations work: shape {stacked.shape}')

    print('')
    print('🎉 All dependencies are working correctly!')

except Exception as e:
    print(f'❌ Error: {e}')
    print('')
    print('❌ Some dependencies are still not working')
    exit(1)
"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Fix completed successfully!"
    echo ""
    echo "You can now run your app with:"
    echo "  source .venv/bin/activate"
    echo "  streamlit run app.py"
    echo ""
    echo "Or use the run script:"
    echo "  ./run_app.sh"
else
    echo ""
    echo "❌ Issues remain. Try these additional steps:"
    echo "1. Delete and recreate the virtual environment:"
    echo "   rm -rf .venv"
    echo "   python3 -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "2. Check your Python installation"
    echo "3. Try using a different Python version"
fi
