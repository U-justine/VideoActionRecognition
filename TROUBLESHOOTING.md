# Troubleshooting Guide: Video Action Recognition

This guide helps resolve common issues with the Video Action Recognition application, particularly the "Numpy is not available" error.

## Quick Fix Instructions

### 1. Fix Numpy Issues (Recommended)

Open Terminal and navigate to your project folder:

```bash
cd "/Users/williammuorwel/Desktop/Video Action Recognition"
```

Run the fix script:
```bash
chmod +x run_fix.sh
./run_fix.sh
```

### 2. Manual Fix Steps

If the script doesn't work, follow these manual steps:

#### Step 1: Activate Virtual Environment
```bash
cd "/Users/williammuorwel/Desktop/Video Action Recognition"
source .venv/bin/activate
```

#### Step 2: Upgrade pip
```bash
python -m pip install --upgrade pip
```

#### Step 3: Reinstall numpy
```bash
python -m pip install --force-reinstall --no-cache-dir "numpy>=1.24.0"
```

#### Step 4: Install other dependencies
```bash
pip install --upgrade "Pillow>=10.0.0"
pip install --upgrade "opencv-python>=4.9.0"
pip install -r requirements.txt
```

#### Step 5: Test numpy
```bash
python -c "import numpy; print(f'Numpy version: {numpy.__version__}')"
```

### 3. Run the Application

After fixing numpy, run the app:

```bash
streamlit run app.py
```

Or use the run script:
```bash
chmod +x run_app.sh
./run_app.sh
```

## Common Error Messages and Solutions

### "Numpy is not available"
**Cause:** Numpy installation is corrupted or missing
**Solution:** Follow the manual fix steps above, especially step 3

### "Unable to process video frames"
**Possible causes:**
- Video file is corrupted or unsupported format
- Numpy operations are failing
- Insufficient memory

**Solutions:**
1. Try a different video file (MP4 recommended)
2. Ensure video is less than 200MB
3. Fix numpy installation (see above)
4. Restart the application

### "ModuleNotFoundError: No module named 'xyz'"
**Cause:** Missing Python package
**Solution:** 
```bash
pip install -r requirements.txt
```

### Virtual Environment Issues
If you get errors about virtual environment:

1. **Recreate virtual environment:**
```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Check Python version:**
```bash
python --version
```
Make sure you have Python 3.8 or higher.

## Video Requirements

### Supported Formats
- MP4 (recommended)
- AVI
- MOV
- MKV

### Recommendations
- File size: Less than 200MB
- Duration: 1-60 seconds
- Resolution: Any (will be resized to 224x224)
- Clear, visible actions work best

### Unsupported
- Audio-only files
- Very long videos (>5 minutes)
- Corrupted files

## Diagnostic Commands

Use these commands to diagnose issues:

### Check Python Environment
```bash
python --version
which python
echo $VIRTUAL_ENV
```

### Test Dependencies
```bash
python -c "import numpy; print('Numpy OK')"
python -c "import torch; print('PyTorch OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "from transformers import AutoImageProcessor; print('Transformers OK')"
```

### Check Video Processing
```bash
python -c "
import numpy as np
from PIL import Image
test_img = Image.new('RGB', (224, 224), 'red')
arr = np.array(test_img, dtype=np.float32)
print(f'Image to array conversion: OK, shape {arr.shape}')
"
```

## Advanced Troubleshooting

### If Nothing Works
1. **Check system requirements:**
   - macOS 10.15 or later
   - Python 3.8 or higher
   - At least 4GB free RAM

2. **Try different Python version:**
```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Clear Python caches:**
```bash
find . -type d -name "__pycache__" -delete
find . -name "*.pyc" -delete
```

4. **Check for conflicting installations:**
```bash
pip list | grep numpy
pip list | grep torch
```

### Performance Issues
- Close other applications to free up memory
- Use shorter videos (< 30 seconds)
- Ensure stable internet connection (for model download)

## Getting Help

If you're still having issues:

1. **Check the error message carefully** - the improved error handling will give you specific guidance
2. **Try the diagnostic commands** above to identify the specific problem
3. **Look at the Terminal output** - it often contains helpful debugging information
4. **Try a different video file** - some files may be corrupted or unsupported

## Model Information

The app uses:
- **Model:** facebook/timesformer-base-finetuned-k400
- **Input:** 8 uniformly sampled frames at 224x224 pixels
- **Actions:** 400+ action classes including sports, cooking, music, dancing, daily activities

First run will download the model (~1.5GB), which requires internet connection.