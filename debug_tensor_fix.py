#!/usr/bin/env python3
"""
Debug script to test and verify the tensor creation fix.
This script isolates the problematic code and tests various scenarios.
"""

import sys
import tempfile
from pathlib import Path
import logging
import numpy as np
from PIL import Image

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_frames(num_frames=8, size=(224, 224)):
    """Create synthetic test frames to simulate video processing."""
    frames = []
    for i in range(num_frames):
        # Create a simple gradient image
        img_array = np.zeros((*size, 3), dtype=np.uint8)

        # Add some variation between frames
        gradient = np.linspace(0, 255, size[0]).astype(np.uint8)
        for j in range(3):  # RGB channels
            img_array[:, :, j] = gradient + (i * 10) % 256

        # Convert to PIL Image
        frame = Image.fromarray(img_array, 'RGB')
        frames.append(frame)

    return frames

def test_processor_approaches():
    """Test different approaches to fix the tensor creation issue."""

    print("🔍 Testing Tensor Creation Fix")
    print("=" * 50)

    try:
        from transformers import AutoImageProcessor, TimesformerForVideoClassification
        import torch
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return False

    # Load processor (but not full model to save time/memory)
    try:
        processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        print("✅ Processor loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load processor: {e}")
        return False

    # Test with different frame scenarios
    test_scenarios = [
        {"name": "Standard 8 frames", "frames": 8, "size": (224, 224)},
        {"name": "Different count (6 frames)", "frames": 6, "size": (224, 224)},
        {"name": "Different size frames", "frames": 8, "size": (256, 256)},
        {"name": "Single frame", "frames": 1, "size": (224, 224)},
    ]

    success_count = 0

    for scenario in test_scenarios:
        print(f"\n📋 Testing: {scenario['name']}")
        print("-" * 30)

        frames = create_test_frames(scenario["frames"], scenario["size"])
        required_frames = 8  # TimeSformer default

        # Apply the same logic as in our fix
        if len(frames) != required_frames:
            print(f"⚠️  Frame count mismatch: {len(frames)} vs {required_frames}")
            if len(frames) < required_frames:
                frames.extend([frames[-1]] * (required_frames - len(frames)))
                print(f"🔧 Padded to {len(frames)} frames")
            else:
                frames = frames[:required_frames]
                print(f"🔧 Truncated to {len(frames)} frames")

        # Ensure consistent frame sizes
        if frames:
            target_size = (224, 224)  # Standard size for TimeSformer
            frames = [frame.resize(target_size) if frame.size != target_size else frame for frame in frames]
            print(f"🔧 Normalized all frames to {target_size}")

        # Test different processor approaches
        approaches = [
            ("Direct with padding", lambda: processor(images=frames, return_tensors="pt", padding=True)),
            ("List wrapped with padding", lambda: processor(images=[frames], return_tensors="pt", padding=True)),
            ("Direct without padding", lambda: processor(images=frames, return_tensors="pt")),
            ("Manual tensor creation", lambda: create_manual_tensor(frames, processor)),
        ]

        for approach_name, approach_func in approaches:
            try:
                print(f"  🧪 Trying: {approach_name}")
                inputs = approach_func()

                # Check tensor properties
                if 'pixel_values' in inputs:
                    tensor = inputs['pixel_values']
                    print(f"    ✅ Success! Tensor shape: {tensor.shape}")
                    print(f"    📊 Tensor dtype: {tensor.dtype}")
                    print(f"    📈 Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
                    success_count += 1
                    break
                else:
                    print(f"    ❌ No pixel_values in output: {inputs.keys()}")

            except Exception as e:
                print(f"    ❌ Failed: {str(e)[:100]}...")
                continue
        else:
            print(f"  💥 All approaches failed for {scenario['name']}")

    print(f"\n📊 Summary: {success_count}/{len(test_scenarios)} scenarios passed")
    return success_count == len(test_scenarios)

def create_manual_tensor(frames, processor):
    """Manual tensor creation as final fallback."""
    if not frames:
        raise ValueError("No frames provided")

    frame_arrays = []
    for frame in frames:
        # Ensure RGB mode
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')
        # Resize to standard size
        frame = frame.resize((224, 224))
        frame_array = np.array(frame)
        frame_arrays.append(frame_array)

    # Stack frames: (num_frames, height, width, channels)
    video_array = np.stack(frame_arrays)

    # Convert to tensor and normalize
    video_tensor = torch.tensor(video_array, dtype=torch.float32) / 255.0

    # Rearrange dimensions for TimeSformer: (batch, channels, num_frames, height, width)
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)

    return {'pixel_values': video_tensor}

def test_video_processing():
    """Test with actual video processing simulation."""
    print(f"\n🎬 Testing Video Processing Pipeline")
    print("=" * 50)

    try:
        # Create a temporary "video" by saving frames as images
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create test frames and save them
            frames = create_test_frames(8, (640, 480))  # Different size to test resizing
            frame_paths = []

            for i, frame in enumerate(frames):
                frame_path = tmp_path / f"frame_{i:03d}.jpg"
                frame.save(frame_path)
                frame_paths.append(frame_path)

            print(f"✅ Created {len(frame_paths)} test frames")

            # Load frames back (simulating video reading)
            loaded_frames = []
            for frame_path in frame_paths:
                frame = Image.open(frame_path)
                loaded_frames.append(frame)

            print(f"✅ Loaded {len(loaded_frames)} frames")

            # Test processing
            return test_single_scenario(loaded_frames, "Video simulation")

    except Exception as e:
        print(f"❌ Video processing test failed: {e}")
        return False

def test_single_scenario(frames, scenario_name):
    """Test a single scenario with comprehensive error handling."""
    print(f"\n🎯 Testing scenario: {scenario_name}")

    try:
        from transformers import AutoImageProcessor
        import torch

        processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

        # Apply our fix logic
        required_frames = 8

        if len(frames) != required_frames:
            if len(frames) < required_frames:
                frames.extend([frames[-1]] * (required_frames - len(frames)))
            else:
                frames = frames[:required_frames]

        # Normalize frame sizes
        target_size = (224, 224)
        frames = [frame.resize(target_size) if frame.size != target_size else frame for frame in frames]

        # Try our primary approach
        inputs = processor(images=frames, return_tensors="pt", padding=True)

        print(f"✅ Success! Tensor shape: {inputs['pixel_values'].shape}")
        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    print("🐛 Tensor Creation Debug Suite")
    print("=" * 60)

    # Test 1: Processor approaches
    test1_passed = test_processor_approaches()

    # Test 2: Video processing simulation
    test2_passed = test_video_processing()

    print(f"\n🏁 Final Results:")
    print(f"   Processor tests: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Video tests: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print(f"\n🎉 All tests passed! The tensor fix should work correctly.")
        sys.exit(0)
    else:
        print(f"\n💥 Some tests failed. Check the logs above for details.")
        sys.exit(1)
