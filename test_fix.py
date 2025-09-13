#!/usr/bin/env python3
"""
Test script to verify the video processing fix works correctly.
This script tests the predict_actions function with different scenarios.
"""

import sys
import tempfile
from pathlib import Path

import logging

# Configure logging to see debug output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from predict import predict_actions, _read_video_frames, load_model
    print("✓ Successfully imported predict functions")
except ImportError as e:
    print(f"✗ Failed to import predict functions: {e}")
    sys.exit(1)

def create_test_video(output_path: Path, duration: int = 2, fps: int = 10):
    """Create a simple test video using OpenCV."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV not available for creating test video")
        return False

    # Create a simple test video with moving rectangle
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (224, 224))

    total_frames = duration * fps
    for i in range(total_frames):
        # Create frame with moving rectangle
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        x_pos = int(50 + 100 * (i / total_frames))
        cv2.rectangle(frame, (x_pos, 50), (x_pos + 50, 150), (0, 255, 0), -1)
        out.write(frame)

    out.release()
    return True

def test_frame_reading(video_path: Path):
    """Test frame reading functionality."""
    print(f"\n--- Testing frame reading from {video_path.name} ---")

    try:
        frames = _read_video_frames(video_path, num_frames=8)
        print(f"✓ Successfully read {len(frames)} frames")

        # Check frame properties
        if frames:
            frame = frames[0]
            print(f"✓ Frame size: {frame.size}")
            print(f"✓ Frame mode: {frame.mode}")

            # Check all frames have same size
            sizes = [f.size for f in frames]
            if len(set(sizes)) == 1:
                print("✓ All frames have consistent size")
            else:
                print(f"⚠ Inconsistent frame sizes: {set(sizes)}")

        return True
    except Exception as e:
        print(f"✗ Frame reading failed: {e}")
        return False

def test_model_loading():
    """Test model loading functionality."""
    print("\n--- Testing model loading ---")

    try:
        processor, model, device = load_model()
        print(f"✓ Successfully loaded model on device: {device}")
        print(f"✓ Model config num_frames: {getattr(model.config, 'num_frames', 'Not specified')}")
        return True, (processor, model, device)
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False, (None, None, None)

def test_prediction(video_path: Path):
    """Test full prediction pipeline."""
    print(f"\n--- Testing prediction on {video_path.name} ---")

    try:
        predictions = predict_actions(str(video_path), top_k=3)
        print(f"✓ Successfully got {len(predictions)} predictions")

        for i, (label, score) in enumerate(predictions, 1):
            print(f"  {i}. {label}: {score:.4f} ({score*100:.2f}%)")

        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Starting Video Action Recognition Test Suite")

    # Test 1: Model loading
    model_loaded, _ = test_model_loading()
    if not model_loaded:
        print("❌ Model loading failed - cannot continue tests")
        return

    # Test 2: Create test video
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_video_path = Path(tmp_dir) / "test_video.mp4"

        print(f"\n--- Creating test video at {test_video_path} ---")
        if create_test_video(test_video_path):
            print("✓ Test video created successfully")

            # Test 3: Frame reading
            if test_frame_reading(test_video_path):
                print("✓ Frame reading test passed")
            else:
                print("❌ Frame reading test failed")
                return

            # Test 4: Full prediction
            if test_prediction(test_video_path):
                print("✅ All tests passed! The fix is working correctly.")
            else:
                print("❌ Prediction test failed")
        else:
            print("⚠ Could not create test video, skipping video-based tests")
            print("💡 Try testing with an existing video file")

if __name__ == "__main__":
    main()
