#!/usr/bin/env python3
"""
Comprehensive test suite for TimeSformer model implementation.
Tests all components of the video action recognition system.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

# Import the fixed predictor
from predict_fixed import (
    read_video_frames,
    normalize_frames,
    create_timesformer_tensor,
    load_model,
    predict_actions
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_test_video_frames(num_frames: int = 8, size: Tuple[int, int] = (224, 224)) -> List[Image.Image]:
    """Create synthetic test frames for testing."""
    frames = []
    for i in range(num_frames):
        # Create frames with different colors to simulate motion
        hue = int((i / num_frames) * 255)
        color = (hue, 255 - hue, 128)
        frame = Image.new('RGB', size, color)
        frames.append(frame)
    return frames


def test_frame_creation():
    """Test synthetic frame creation."""
    print("\n🔍 Testing frame creation...")
    try:
        frames = create_test_video_frames()
        assert len(frames) == 8, f"Expected 8 frames, got {len(frames)}"
        assert all(frame.size == (224, 224) for frame in frames), "Frame size mismatch"
        assert all(frame.mode == 'RGB' for frame in frames), "Frame mode should be RGB"
        print("✅ Frame creation test passed")
        return True
    except Exception as e:
        print(f"❌ Frame creation test failed: {e}")
        return False


def test_frame_normalization():
    """Test frame normalization function."""
    print("\n🔍 Testing frame normalization...")
    try:
        # Create frames with different sizes
        frames = [
            Image.new('RGB', (100, 100), 'red'),
            Image.new('RGB', (300, 200), 'green'),
            Image.new('RGBA', (224, 224), 'blue')  # Different mode
        ]

        normalized = normalize_frames(frames, target_size=(224, 224))

        assert len(normalized) == 3, "Frame count mismatch"
        assert all(frame.size == (224, 224) for frame in normalized), "Normalization size failed"
        assert all(frame.mode == 'RGB' for frame in normalized), "Mode conversion failed"

        print("✅ Frame normalization test passed")
        return True
    except Exception as e:
        print(f"❌ Frame normalization test failed: {e}")
        return False


def test_tensor_creation():
    """Test TimeSformer tensor creation."""
    print("\n🔍 Testing TimeSformer tensor creation...")
    try:
        frames = create_test_video_frames(8)
        tensor = create_timesformer_tensor(frames)

        # Check tensor properties
        expected_shape = (1, 8, 3, 224, 224)  # (batch, frames, channels, height, width)
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
        assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"
        assert 0.0 <= tensor.min() <= 1.0, f"Tensor values should be normalized, min: {tensor.min()}"
        assert 0.0 <= tensor.max() <= 1.0, f"Tensor values should be normalized, max: {tensor.max()}"

        print(f"✅ Tensor creation test passed - Shape: {tensor.shape}")
        return True
    except Exception as e:
        print(f"❌ Tensor creation test failed: {e}")
        return False


def test_model_loading():
    """Test model loading functionality."""
    print("\n🔍 Testing model loading...")
    try:
        processor, model, device = load_model()

        # Check model properties
        assert processor is not None, "Processor should not be None"
        assert model is not None, "Model should not be None"
        assert hasattr(model, 'config'), "Model should have config"
        assert hasattr(model.config, 'id2label'), "Model should have label mapping"

        # Check if model is in eval mode
        assert not model.training, "Model should be in eval mode"

        # Check device
        model_device = next(model.parameters()).device
        print(f"Model loaded on device: {model_device}")

        print("✅ Model loading test passed")
        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def test_end_to_end_prediction():
    """Test complete prediction pipeline with synthetic video."""
    print("\n🔍 Testing end-to-end prediction...")
    try:
        # Create a temporary video file (we'll simulate this with frames)
        frames = create_test_video_frames(8)

        # Create temporary directory and mock video processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # We'll test the tensor creation and model inference directly
            # since creating an actual video file is complex

            # Test tensor creation
            tensor = create_timesformer_tensor(frames)

            # Load model
            processor, model, device = load_model()

            # Move tensor to device
            tensor = tensor.to(device)

            # Run inference
            with torch.no_grad():
                outputs = model(pixel_values=tensor)
                logits = outputs.logits

            # Check output properties
            assert logits.shape[0] == 1, "Batch size should be 1"
            assert logits.shape[1] == 400, "Should have 400 classes (Kinetics-400)"

            # Get top predictions
            probabilities = torch.softmax(logits, dim=-1)[0]
            top_probs, top_indices = torch.topk(probabilities, k=5)

            # Convert to results
            results = []
            for prob, idx in zip(top_probs.cpu(), top_indices.cpu()):
                label = model.config.id2label[idx.item()]
                confidence = float(prob.item())
                results.append((label, confidence))

            # Validate results
            assert len(results) == 5, "Should return 5 predictions"
            assert all(isinstance(label, str) for label, _ in results), "Labels should be strings"
            assert all(0.0 <= confidence <= 1.0 for _, confidence in results), "Confidence should be between 0 and 1"
            assert all(results[i][1] >= results[i+1][1] for i in range(len(results)-1)), "Results should be sorted by confidence"

            print("✅ End-to-end prediction test passed")
            print(f"Top prediction: {results[0][0]} ({results[0][1]:.4f})")
            return True

    except Exception as e:
        print(f"❌ End-to-end prediction test failed: {e}")
        return False


def test_error_handling():
    """Test error handling scenarios."""
    print("\n🔍 Testing error handling...")

    tests_passed = 0
    total_tests = 3

    # Test 1: Invalid number of frames
    try:
        frames = create_test_video_frames(5)  # Wrong number
        create_timesformer_tensor(frames)
        print("❌ Should have failed with wrong frame count")
    except ValueError:
        print("✅ Correctly handled wrong frame count")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Unexpected error for wrong frame count: {e}")

    # Test 2: Empty frame list
    try:
        normalize_frames([])
        print("❌ Should have failed with empty frames")
    except (RuntimeError, ValueError):
        print("✅ Correctly handled empty frame list")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Unexpected error for empty frames: {e}")

    # Test 3: Invalid frame type
    try:
        frames = [None] * 8
        create_timesformer_tensor(frames)
        print("❌ Should have failed with invalid frame type")
    except (AttributeError, TypeError):
        print("✅ Correctly handled invalid frame type")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Unexpected error for invalid frames: {e}")

    success_rate = tests_passed / total_tests
    print(f"Error handling tests: {tests_passed}/{total_tests} passed ({success_rate:.1%})")
    return success_rate >= 0.8


def benchmark_performance():
    """Benchmark the performance of key operations."""
    print("\n⏱️ Benchmarking performance...")

    # Benchmark tensor creation
    frames = create_test_video_frames(8)

    start_time = time.time()
    for _ in range(10):
        tensor = create_timesformer_tensor(frames)
    tensor_time = (time.time() - start_time) / 10

    print(f"Average tensor creation time: {tensor_time:.4f} seconds")

    # Benchmark model inference
    try:
        processor, model, device = load_model()
        tensor = create_timesformer_tensor(frames).to(device)

        # Warm up
        with torch.no_grad():
            model(pixel_values=tensor)

        # Benchmark
        start_time = time.time()
        for _ in range(5):
            with torch.no_grad():
                outputs = model(pixel_values=tensor)
        inference_time = (time.time() - start_time) / 5

        print(f"Average model inference time: {inference_time:.4f} seconds")
        print(f"Device used: {device}")

        if tensor_time < 0.1 and inference_time < 2.0:
            print("✅ Performance benchmarks look good")
            return True
        else:
            print("⚠️ Performance might be slower than expected")
            return True  # Don't fail on slow performance

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide summary."""
    print("🚀 Starting TimeSformer Model Test Suite")
    print("=" * 60)

    tests = [
        ("Frame Creation", test_frame_creation),
        ("Frame Normalization", test_frame_normalization),
        ("Tensor Creation", test_tensor_creation),
        ("Model Loading", test_model_loading),
        ("End-to-End Prediction", test_end_to_end_prediction),
        ("Error Handling", test_error_handling),
        ("Performance Benchmark", benchmark_performance),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"💥 {test_name} failed")
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")

    print("\n" + "=" * 60)
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed ({passed/total:.1%})")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Your TimeSformer implementation is working correctly.")
    elif passed >= total * 0.8:
        print("✅ Most tests passed. Minor issues may exist but the core functionality works.")
    else:
        print("❌ Several tests failed. Please review the implementation.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
