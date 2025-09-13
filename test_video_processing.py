#!/usr/bin/env python3
"""
Test script to verify video processing functionality.
Creates a synthetic test video and tests the prediction pipeline.
"""

import sys
import tempfile
import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_synthetic_video(output_path: Path, duration_seconds: float = 2.0, fps: int = 24):
    """Create a synthetic test video with simple animation."""

    width, height = 640, 480
    total_frames = int(duration_seconds * fps)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    logging.info(f"Creating synthetic video: {total_frames} frames at {fps} FPS")

    for frame_num in range(total_frames):
        # Create a frame with animated content
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add background gradient
        for y in range(height):
            intensity = int(255 * (y / height))
            frame[y, :] = [intensity // 3, intensity // 2, intensity]

        # Add moving circle (simulating an action)
        center_x = int(width * (0.2 + 0.6 * frame_num / total_frames))
        center_y = height // 2
        radius = 30 + int(20 * np.sin(frame_num * 0.3))

        # Convert to PIL for drawing
        pil_frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_frame)

        # Draw moving circle
        left = center_x - radius
        top = center_y - radius
        right = center_x + radius
        bottom = center_y + radius
        draw.ellipse([left, top, right, bottom], fill=(255, 255, 0))

        # Add some text to simulate action
        draw.text((50, 50), f"Frame {frame_num}", fill=(255, 255, 255))
        draw.text((50, 80), "Synthetic Action", fill=(255, 255, 255))

        # Convert back to numpy and BGR for OpenCV
        frame = np.array(pil_frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(frame_bgr)

    out.release()
    logging.info(f"✓ Created synthetic video: {output_path}")
    return output_path

def test_video_reading():
    """Test video reading functionality without full model inference."""

    logging.info("=== Testing Video Reading ===")

    try:
        from predict import _read_video_frames, normalize_frames

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = tmp_path / "test_video.mp4"

            # Create test video
            create_synthetic_video(video_path, duration_seconds=1.0, fps=12)  # Short video

            # Test reading frames
            logging.info("Testing frame reading...")
            frames = _read_video_frames(video_path, num_frames=8)

            if not frames:
                logging.error("✗ No frames extracted")
                return False

            logging.info(f"✓ Extracted {len(frames)} frames")

            # Test frame normalization
            logging.info("Testing frame normalization...")
            normalized = normalize_frames(frames, required_frames=8)

            if len(normalized) != 8:
                logging.error(f"✗ Expected 8 frames, got {len(normalized)}")
                return False

            logging.info("✓ Frame normalization successful")

            # Check frame properties
            for i, frame in enumerate(normalized):
                if frame.size != (224, 224):
                    logging.error(f"✗ Frame {i} has wrong size: {frame.size}")
                    return False
                if frame.mode != 'RGB':
                    logging.error(f"✗ Frame {i} has wrong mode: {frame.mode}")
                    return False

            logging.info("✓ All frames have correct properties")
            return True

    except Exception as e:
        logging.error(f"✗ Video reading test failed: {e}")
        return False

def test_tensor_creation():
    """Test tensor creation from frames."""

    logging.info("=== Testing Tensor Creation ===")

    try:
        from predict import create_tensor_from_frames
        import torch

        # Create dummy frames
        frames = []
        for i in range(8):
            frame = Image.new('RGB', (224, 224), (i*30 % 255, 100, 150))
            frames.append(frame)

        logging.info("Testing tensor creation...")
        tensor = create_tensor_from_frames(frames, processor=None)  # Use manual creation

        # Check tensor properties
        expected_shape = (1, 3, 8, 224, 224)  # (batch, channels, frames, height, width)
        if tensor.shape != expected_shape:
            logging.error(f"✗ Expected shape {expected_shape}, got {tensor.shape}")
            return False

        logging.info(f"✓ Tensor created with correct shape: {tensor.shape}")

        # Check tensor values are in reasonable range
        if tensor.min() < 0 or tensor.max() > 1:
            logging.warning(f"⚠ Tensor values outside [0,1]: [{tensor.min():.3f}, {tensor.max():.3f}]")

        logging.info("✓ Tensor creation successful")
        return True

    except Exception as e:
        logging.error(f"✗ Tensor creation test failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete prediction pipeline with a synthetic video."""

    logging.info("=== Testing Full Pipeline ===")

    try:
        from predict import predict_actions

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = tmp_path / "test_video.mp4"

            # Create test video
            create_synthetic_video(video_path, duration_seconds=2.0, fps=15)

            logging.info("Running full prediction pipeline...")

            # Run prediction with smaller top_k for faster testing
            results = predict_actions(str(video_path), top_k=3)

            if not results:
                logging.error("✗ No predictions returned")
                return False

            logging.info(f"✓ Got {len(results)} predictions")

            # Display results
            for i, (label, confidence) in enumerate(results, 1):
                logging.info(f"  {i}. {label}: {confidence:.3f}")

            # Basic validation
            if len(results) != 3:
                logging.error(f"✗ Expected 3 results, got {len(results)}")
                return False

            for label, confidence in results:
                if not isinstance(label, str) or not isinstance(confidence, float):
                    logging.error(f"✗ Invalid result format: {label}, {confidence}")
                    return False
                if confidence < 0 or confidence > 1:
                    logging.error(f"✗ Invalid confidence: {confidence}")
                    return False

            logging.info("✓ Full pipeline test successful")
            return True

    except Exception as e:
        logging.error(f"✗ Full pipeline test failed: {e}")
        logging.exception("Full error traceback:")
        return False

def main():
    """Run all tests."""

    print("🧪 Video Processing Test Suite")
    print("=" * 50)

    tests = [
        ("Video Reading", test_video_reading),
        ("Tensor Creation", test_tensor_creation),
        ("Full Pipeline", test_full_pipeline),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)

        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            logging.exception(f"Test {test_name} crashed:")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Video processing is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())
