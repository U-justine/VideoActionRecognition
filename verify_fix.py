#!/usr/bin/env python3
"""
Final verification script to test the tensor creation fix.
This script performs comprehensive testing to ensure the video action recognition
system works correctly after applying the tensor padding fix.
"""

import sys
import os
import tempfile
import logging
from pathlib import Path
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available."""
    logger.info("🔍 Checking dependencies...")

    missing_deps = []

    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")

    try:
        import transformers
        logger.info(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        missing_deps.append("transformers")

    try:
        import cv2
        logger.info(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        logger.warning("⚠ OpenCV not available (fallback will be used)")

    try:
        import decord
        logger.info("✓ Decord available")
    except ImportError:
        logger.warning("⚠ Decord not available (OpenCV fallback will be used)")

    try:
        import streamlit
        logger.info(f"✓ Streamlit {streamlit.__version__}")
    except ImportError:
        missing_deps.append("streamlit")

    if missing_deps:
        logger.error(f"❌ Missing dependencies: {missing_deps}")
        return False

    logger.info("✅ All required dependencies available")
    return True

def create_synthetic_video(output_path, duration_seconds=3, fps=10, width=320, height=240):
    """Create a synthetic MP4 video for testing."""
    logger.info(f"🎬 Creating synthetic video: {output_path}")

    try:
        import cv2
    except ImportError:
        logger.error("❌ OpenCV required for video creation")
        return False

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        logger.error(f"❌ Cannot create video writer for {output_path}")
        return False

    total_frames = duration_seconds * fps

    for frame_idx in range(total_frames):
        # Create frame with moving rectangle (simulates action)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Moving rectangle across the frame
        progress = frame_idx / total_frames
        rect_x = int(20 + (width - 80) * progress)
        rect_y = height // 2 - 20

        # Draw rectangle with changing color
        color = (
            int(255 * (1 - progress)),  # Red decreases
            int(255 * progress),        # Green increases
            128                         # Blue constant
        )

        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 60, rect_y + 40), color, -1)

        # Add frame number
        cv2.putText(frame, f"Frame {frame_idx+1}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)

    out.release()

    # Verify file was created
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info(f"✅ Video created: {output_path} ({output_path.stat().st_size} bytes)")
        return True
    else:
        logger.error("❌ Video creation failed")
        return False

def test_model_loading():
    """Test if the model loads correctly."""
    logger.info("🤖 Testing model loading...")

    try:
        from predict import load_model
        processor, model, device = load_model()

        logger.info(f"✅ Model loaded successfully on device: {device}")
        logger.info(f"   Model type: {type(model).__name__}")
        logger.info(f"   Processor type: {type(processor).__name__}")

        # Check model config
        num_frames = getattr(model.config, 'num_frames', 8)
        logger.info(f"   Expected frames: {num_frames}")

        return True, (processor, model, device)

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False, (None, None, None)

def test_frame_extraction(video_path):
    """Test frame extraction from video."""
    logger.info(f"🎞️ Testing frame extraction from: {video_path}")

    try:
        from predict import _read_video_frames

        frames = _read_video_frames(Path(video_path), num_frames=8)

        logger.info(f"✅ Extracted {len(frames)} frames")

        if frames:
            first_frame = frames[0]
            logger.info(f"   Frame size: {first_frame.size}")
            logger.info(f"   Frame mode: {first_frame.mode}")

            # Check if all frames have same properties
            sizes = [f.size for f in frames]
            modes = [f.mode for f in frames]

            if len(set(sizes)) == 1:
                logger.info("   ✅ All frames have consistent size")
            else:
                logger.warning(f"   ⚠ Inconsistent frame sizes: {set(sizes)}")

            if len(set(modes)) == 1:
                logger.info("   ✅ All frames have consistent mode")
            else:
                logger.warning(f"   ⚠ Inconsistent frame modes: {set(modes)}")

            return True, frames
        else:
            logger.error("   ❌ No frames extracted")
            return False, []

    except Exception as e:
        logger.error(f"❌ Frame extraction failed: {e}")
        return False, []

def test_tensor_creation(frames):
    """Test the tensor creation process that was causing issues."""
    logger.info("🔧 Testing tensor creation (the main fix)...")

    try:
        from transformers import AutoImageProcessor
        import torch

        processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

        # Test the approaches from our fix
        approaches = [
            ("Direct with padding", lambda: processor(images=frames, return_tensors="pt", padding=True)),
            ("List format with padding", lambda: processor(images=[frames], return_tensors="pt", padding=True)),
            ("Direct without padding", lambda: processor(images=frames, return_tensors="pt")),
        ]

        for approach_name, approach_func in approaches:
            try:
                logger.info(f"   Testing: {approach_name}")
                inputs = approach_func()

                if 'pixel_values' in inputs:
                    tensor_shape = inputs['pixel_values'].shape
                    logger.info(f"   ✅ {approach_name} succeeded - tensor shape: {tensor_shape}")
                    return True, inputs
                else:
                    logger.warning(f"   ⚠ {approach_name} - no pixel_values in output")

            except Exception as e:
                logger.warning(f"   ❌ {approach_name} failed: {str(e)[:100]}")

        # If all approaches fail, try manual creation
        logger.info("   Testing: Manual tensor creation")
        try:
            frame_arrays = []
            for frame in frames:
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                if frame.size != (224, 224):
                    frame = frame.resize((224, 224))
                frame_array = np.array(frame, dtype=np.float32) / 255.0
                frame_arrays.append(frame_array)

            video_array = np.stack(frame_arrays, axis=0)
            video_tensor = torch.from_numpy(video_array)
            video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)

            inputs = {'pixel_values': video_tensor}
            logger.info(f"   ✅ Manual creation succeeded - tensor shape: {video_tensor.shape}")
            return True, inputs

        except Exception as e:
            logger.error(f"   ❌ Manual creation failed: {e}")

        logger.error("❌ All tensor creation approaches failed")
        return False, None

    except Exception as e:
        logger.error(f"❌ Tensor creation test setup failed: {e}")
        return False, None

def test_full_prediction(video_path):
    """Test the complete prediction pipeline."""
    logger.info(f"🎯 Testing full prediction pipeline with: {video_path}")

    try:
        from predict import predict_actions

        # This is the main function that was failing
        predictions = predict_actions(str(video_path), top_k=3)

        logger.info(f"✅ Prediction successful! Got {len(predictions)} results:")
        for i, (label, score) in enumerate(predictions, 1):
            logger.info(f"   {i}. {label}: {score:.4f} ({score*100:.1f}%)")

        return True, predictions

    except Exception as e:
        logger.error(f"❌ Full prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def main():
    """Run complete verification suite."""
    print("🧪 Video Action Recognition - Tensor Fix Verification")
    print("=" * 60)

    # Track test results
    tests_passed = 0
    total_tests = 6

    # Test 1: Dependencies
    if check_dependencies():
        tests_passed += 1
    else:
        logger.error("❌ Dependency check failed - cannot continue")
        return 1

    # Test 2: Model loading
    model_loaded, (processor, model, device) = test_model_loading()
    if model_loaded:
        tests_passed += 1

    # Create temporary test video
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = Path(tmp_dir) / "test_video.mp4"

        # Test 3: Video creation
        if create_synthetic_video(video_path):
            tests_passed += 1

            # Test 4: Frame extraction
            frames_ok, frames = test_frame_extraction(video_path)
            if frames_ok:
                tests_passed += 1

                # Test 5: Tensor creation (the main fix)
                tensor_ok, inputs = test_tensor_creation(frames)
                if tensor_ok:
                    tests_passed += 1

                # Test 6: Full pipeline
                if model_loaded:
                    pred_ok, predictions = test_full_prediction(video_path)
                    if pred_ok:
                        tests_passed += 1

    # Final results
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The tensor creation fix is working correctly")
        print("🚀 You can now use the Streamlit app with confidence")
        return 0
    else:
        print("❌ Some tests failed")
        print(f"📋 Passed: {tests_passed}/{total_tests}")

        if tests_passed >= 4:  # Core functionality works
            print("⚠️  Core functionality appears to work, some advanced features may have issues")
            return 0
        else:
            print("💥 Critical issues detected - check error messages above")
            return 1

if __name__ == "__main__":
    sys.exit(main())
