#!/usr/bin/env python3
"""
Diagnostic script to check numpy installation and functionality.
This helps troubleshoot the "Numpy is not available" error.
"""

import sys
import traceback

def check_numpy_import():
    """Check if numpy can be imported."""
    try:
        import numpy as np
        print(f"✓ Numpy imported successfully")
        print(f"✓ Numpy version: {np.__version__}")
        return np
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error importing numpy: {e}")
        traceback.print_exc()
        return None

def check_numpy_basic_operations(np):
    """Test basic numpy operations."""
    if np is None:
        return False

    try:
        # Test array creation
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✓ Array creation works: {arr}")

        # Test array operations
        result = arr * 2
        print(f"✓ Array operations work: {result}")

        # Test float32 arrays (used in the video processing)
        float_arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        print(f"✓ Float32 arrays work: {float_arr}")

        # Test stack operation (used in video processing)
        stacked = np.stack([float_arr, float_arr], axis=0)
        print(f"✓ Stack operation works, shape: {stacked.shape}")

        return True

    except Exception as e:
        print(f"✗ Numpy basic operations failed: {e}")
        traceback.print_exc()
        return False

def check_numpy_with_pil():
    """Test numpy integration with PIL (used in video processing)."""
    try:
        import numpy as np
        from PIL import Image

        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='red')
        print(f"✓ PIL Image created: {test_image}")

        # Convert to numpy array (this is what fails in video processing)
        frame_array = np.array(test_image, dtype=np.float32) / 255.0
        print(f"✓ PIL to numpy conversion works, shape: {frame_array.shape}")

        # Test the exact operation from the video processing code
        frame_arrays = [frame_array, frame_array, frame_array]
        video_array = np.stack(frame_arrays, axis=0)
        print(f"✓ Video array stacking works, shape: {video_array.shape}")

        return True

    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"✗ PIL-numpy integration failed: {e}")
        traceback.print_exc()
        return False

def check_torch_numpy_integration():
    """Test numpy integration with PyTorch."""
    try:
        import numpy as np
        import torch

        # Create numpy array
        np_array = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
        print(f"✓ Numpy array created: shape {np_array.shape}")

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(np_array)
        print(f"✓ Torch tensor from numpy: shape {tensor.shape}")

        # Test permute operation (used in video processing)
        permuted = tensor.permute(2, 0, 1)
        print(f"✓ Tensor permute works: shape {permuted.shape}")

        return True

    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"✗ PyTorch-numpy integration failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic checks."""
    print("=== Numpy Diagnostic Check ===\n")

    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}\n")

    # Check numpy import
    print("1. Checking numpy import...")
    np = check_numpy_import()
    print()

    # Check basic operations
    print("2. Checking basic numpy operations...")
    basic_ok = check_numpy_basic_operations(np)
    print()

    # Check PIL integration
    print("3. Checking PIL-numpy integration...")
    pil_ok = check_numpy_with_pil()
    print()

    # Check PyTorch integration
    print("4. Checking PyTorch-numpy integration...")
    torch_ok = check_torch_numpy_integration()
    print()

    # Summary
    print("=== Summary ===")
    if np is not None and basic_ok and pil_ok and torch_ok:
        print("✓ All checks passed! Numpy should work correctly.")
    else:
        print("✗ Some checks failed. This may explain the 'Numpy is not available' error.")

        # Provide troubleshooting suggestions
        print("\n=== Troubleshooting Suggestions ===")
        if np is None:
            print("- Reinstall numpy: pip install --force-reinstall numpy")
        if not basic_ok:
            print("- Numpy installation may be corrupted")
        if not pil_ok:
            print("- Check PIL/Pillow installation: pip install --upgrade Pillow")
        if not torch_ok:
            print("- Check PyTorch installation: pip install --upgrade torch")

        print("- Try recreating your virtual environment")
        print("- Check for conflicting package versions")

if __name__ == "__main__":
    main()
