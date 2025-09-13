#!/bin/bash

# Script to properly run the Video Action Recognition Streamlit app
# This handles virtual environment activation and dependency checks

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🎬 Video Action Recognition App"
echo "==============================="
echo "Working directory: $SCRIPT_DIR"
echo ""

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [[ ! -d ".venv" ]]; then
    echo "❌ Virtual environment not found"
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to create virtual environment"
        echo "Please ensure Python 3 is installed"
        exit 1
    fi
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ".venv/bin/activate"

if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Failed to activate virtual environment"
    echo "Try running manually:"
    echo "  source .venv/bin/activate"
    echo "  streamlit run app.py"
    exit 1
fi

echo "✅ Virtual environment activated"

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import numpy, torch, transformers, streamlit, cv2" 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "⚠️  Some dependencies missing, installing..."
    pip install -r requirements.txt
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to install dependencies"
        echo "Try running the fix script first: ./run_fix.sh"
        exit 1
    fi
fi

# Final dependency check
echo "Verifying numpy availability..."
python -c "
import numpy as np
print(f'✅ Numpy version: {np.__version__}')

# Test the specific operations used in video processing
try:
    test_array = np.array([[[1, 2, 3]]], dtype=np.float32)
    stacked = np.stack([test_array, test_array], axis=0)
    print('✅ Numpy operations work correctly')
except Exception as e:
    print(f'❌ Numpy operations failed: {e}')
    print('Run the fix script: ./run_fix.sh')
    exit(1)
" 2>/dev/null

if [[ $? -ne 0 ]]; then
    echo "❌ Numpy issues detected"
    echo "Please run the fix script first:"
    echo "  ./run_fix.sh"
    exit 1
fi

echo ""
echo "🚀 Starting Streamlit app..."
echo "The app will open in your default browser"
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Streamlit app
streamlit run app.py

# Deactivate virtual environment when done
deactivate
