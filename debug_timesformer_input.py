#!/usr/bin/env python3
"""
Debug script to understand the expected tensor format for TimeSformer model.
This script tests different tensor shapes and formats to find the correct one.
"""

import torch
import numpy as np
from PIL import Image
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_frames(num_frames=8, size=(224, 224)):
    """Create test frames with different colors to help debug."""
    frames = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 255, 255)]

    for i in range(num_frames):
        color = colors[i % len(colors)]
        frame = Image.new('RGB', size, color)
        frames.append(frame)

    return frames

def test_tensor_shapes():
    """Test different tensor shapes to see what TimeSformer expects."""

    print("🔍 Testing TimeSformer Input Formats")
    print("=" * 50)

    try:
        from transformers import AutoImageProcessor, TimesformerForVideoClassification

        # Load model and processor
        print("Loading TimeSformer model...")
        processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
        model.eval()

        print("✅ Model loaded successfully")
        print(f"Model config num_frames: {getattr(model.config, 'num_frames', 'Not found')}")
        print(f"Model config image_size: {getattr(model.config, 'image_size', 'Not found')}")

        # Create test frames
        frames = create_test_frames(8, (224, 224))
        print(f"✅ Created {len(frames)} test frames")

        # Test 1: Try to use processor (the "correct" way)
        print("\n📋 Test 1: Using Processor")
        try:
            # Different processor approaches
            processor_tests = [
                ("Direct frames", lambda: processor(images=frames, return_tensors="pt")),
                ("List of frames", lambda: processor(images=[frames], return_tensors="pt")),
                ("Videos parameter", lambda: processor(videos=frames, return_tensors="pt") if hasattr(processor, 'videos') else None),
                ("Videos list parameter", lambda: processor(videos=[frames], return_tensors="pt") if hasattr(processor, 'videos') else None),
            ]

            for test_name, test_func in processor_tests:
                try:
                    if test_func is None:
                        continue
                    result = test_func()
                    if result and 'pixel_values' in result:
                        tensor = result['pixel_values']
                        print(f"  ✅ {test_name}: shape {tensor.shape}, dtype {tensor.dtype}, range [{tensor.min():.3f}, {tensor.max():.3f}]")

                        # Try inference with this tensor
                        try:
                            with torch.no_grad():
                                output = model(pixel_values=tensor)
                            print(f"    🎯 Inference successful! Output shape: {output.logits.shape}")
                            return tensor  # Found working format!
                        except Exception as inference_error:
                            print(f"    ❌ Inference failed: {str(inference_error)[:100]}...")
                    else:
                        print(f"  ❌ {test_name}: No pixel_values in result")
                except Exception as e:
                    print(f"  ❌ {test_name}: {str(e)[:100]}...")

        except Exception as e:
            print(f"❌ Processor tests failed: {e}")

        # Test 2: Manual tensor creation with different formats
        print("\n📋 Test 2: Manual Tensor Creation")

        # Convert frames to numpy first
        frame_arrays = []
        for frame in frames:
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            if frame.size != (224, 224):
                frame = frame.resize((224, 224), Image.Resampling.LANCZOS)

            # Convert to numpy array
            frame_array = np.array(frame, dtype=np.float32) / 255.0
            frame_arrays.append(frame_array)

        print(f"Frame arrays created: {len(frame_arrays)} frames of shape {frame_arrays[0].shape}")

        # Test different tensor arrangements
        tensor_tests = [
            # Format: (description, creation_function)
            ("NCHW format", lambda: create_nchw_tensor(frame_arrays)),
            ("NTHW format", lambda: create_nthw_tensor(frame_arrays)),
            ("CTHW format", lambda: create_cthw_tensor(frame_arrays)),
            ("TCHW format", lambda: create_tchw_tensor(frame_arrays)),
            ("Reshaped format", lambda: create_reshaped_tensor(frame_arrays)),
        ]

        for test_name, create_func in tensor_tests:
            try:
                tensor = create_func()
                print(f"  📊 {test_name}: shape {tensor.shape}, dtype {tensor.dtype}")

                # Try inference
                try:
                    with torch.no_grad():
                        output = model(pixel_values=tensor)
                    print(f"    ✅ Inference successful! Output logits shape: {output.logits.shape}")

                    # Get top prediction
                    probs = torch.softmax(output.logits, dim=-1)
                    top_prob, top_idx = torch.max(probs, dim=-1)
                    label = model.config.id2label[top_idx.item()]
                    print(f"    🎯 Top prediction: {label} ({top_prob.item():.3f})")
                    return tensor  # Found working format!

                except Exception as inference_error:
                    error_msg = str(inference_error)
                    if "channels" in error_msg:
                        print(f"    ❌ Channel dimension error: {error_msg[:150]}...")
                    elif "shape" in error_msg:
                        print(f"    ❌ Shape error: {error_msg[:150]}...")
                    else:
                        print(f"    ❌ Inference error: {error_msg[:150]}...")

            except Exception as creation_error:
                print(f"  ❌ {test_name}: Creation failed - {creation_error}")

        print("\n💥 No working tensor format found!")
        return None

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def create_nchw_tensor(frame_arrays):
    """Create tensor in NCHW format: (batch, channels, height, width) for each frame."""
    # This treats each frame independently
    batch_tensors = []
    for frame_array in frame_arrays:
        # frame_array shape: (224, 224, 3)
        frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1)  # (3, 224, 224)
        batch_tensors.append(frame_tensor)

    # Stack into batch: (num_frames, 3, 224, 224)
    return torch.stack(batch_tensors).unsqueeze(0)  # (1, num_frames, 3, 224, 224)

def create_nthw_tensor(frame_arrays):
    """Create tensor in NTHW format: (batch, frames, height, width) - flattened channels."""
    video_array = np.stack(frame_arrays, axis=0)  # (8, 224, 224, 3)
    video_tensor = torch.from_numpy(video_array)
    # Flatten the channel dimension into the frame dimension
    return video_tensor.view(1, 8 * 3, 224, 224)  # (1, 24, 224, 224)

def create_cthw_tensor(frame_arrays):
    """Create tensor in CTHW format: (channels, frames, height, width)."""
    video_array = np.stack(frame_arrays, axis=0)  # (8, 224, 224, 3)
    video_tensor = torch.from_numpy(video_array)
    # Permute to (channels, frames, height, width)
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (3, 8, 224, 224)
    return video_tensor.unsqueeze(0)  # (1, 3, 8, 224, 224)

def create_tchw_tensor(frame_arrays):
    """Create tensor in TCHW format: (frames, channels, height, width)."""
    video_array = np.stack(frame_arrays, axis=0)  # (8, 224, 224, 3)
    video_tensor = torch.from_numpy(video_array)
    # Permute to (frames, channels, height, width)
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # (8, 3, 224, 224)
    return video_tensor.unsqueeze(0)  # (1, 8, 3, 224, 224)

def create_reshaped_tensor(frame_arrays):
    """Try reshaping the tensor completely."""
    video_array = np.stack(frame_arrays, axis=0)  # (8, 224, 224, 3)
    video_tensor = torch.from_numpy(video_array)

    # Try different reshape approaches
    total_elements = video_tensor.numel()

    # Approach: Treat the entire video as one big image with multiple channels
    # Reshape to (1, 3*8, 224, 224) = (1, 24, 224, 224)
    return video_tensor.permute(3, 0, 1, 2).contiguous().view(1, 3*8, 224, 224)

def test_working_examples():
    """Test with known working examples from other implementations."""

    print("\n🔬 Testing Known Working Examples")
    print("=" * 40)

    try:
        # Create a tensor that should definitely work based on the error messages we've seen
        # The model expects input[3, 8, 224, 224] but we keep giving it something else

        # Let's create exactly what the error message suggests
        test_tensor = torch.randn(1, 3, 8, 224, 224)  # Random tensor with exact expected shape
        print(f"Random tensor shape: {test_tensor.shape}")

        from transformers import TimesformerForVideoClassification
        model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

        try:
            with torch.no_grad():
                output = model(pixel_values=test_tensor)
            print(f"✅ Random tensor inference successful! Output shape: {output.logits.shape}")

            # Now we know the format works, let's create real data in this format
            frames = create_test_frames(8, (224, 224))

            # Create tensor in the exact same format as the random one that worked
            frame_tensors = []
            for frame in frames:
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                if frame.size != (224, 224):
                    frame = frame.resize((224, 224), Image.Resampling.LANCZOS)

                # Convert to tensor: (height, width, channels) -> (channels, height, width)
                frame_array = np.array(frame, dtype=np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1)  # (3, 224, 224)
                frame_tensors.append(frame_tensor)

            # Stack channels first, then frames: (3, 8, 224, 224)
            # We want: batch=1, channels=3, frames=8, height=224, width=224
            channel_tensors = []
            for c in range(3):  # For each color channel
                channel_frames = []
                for frame_tensor in frame_tensors:  # For each frame
                    channel_frames.append(frame_tensor[c])  # Get this channel
                channel_tensor = torch.stack(channel_frames)  # (8, 224, 224)
                channel_tensors.append(channel_tensor)

            final_tensor = torch.stack(channel_tensors).unsqueeze(0)  # (1, 3, 8, 224, 224)
            print(f"Real data tensor shape: {final_tensor.shape}")

            # Test inference with real data
            with torch.no_grad():
                output = model(pixel_values=final_tensor)
            print(f"✅ Real data inference successful!")

            # Get prediction
            probs = torch.softmax(output.logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=3, dim=-1)

            print("🎯 Top 3 predictions:")
            for i in range(3):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                label = model.config.id2label[idx]
                print(f"   {i+1}. {label}: {prob:.3f}")

            return final_tensor

        except Exception as e:
            print(f"❌ Even random tensor failed: {e}")

    except Exception as e:
        print(f"❌ Known examples test failed: {e}")

    return None

def main():
    """Run all debug tests."""

    print("🐛 TimeSformer Input Format Debug")
    print("=" * 60)

    # Test 1: Standard approaches
    working_tensor = test_tensor_shapes()

    if working_tensor is not None:
        print(f"\n🎉 Found working tensor format: {working_tensor.shape}")
        return 0

    # Test 2: Known working examples
    working_tensor = test_working_examples()

    if working_tensor is not None:
        print(f"\n🎉 Found working tensor format: {working_tensor.shape}")
        return 0

    print("\n💥 No working tensor format found. This suggests a deeper compatibility issue.")
    print("\n🔧 Recommendations:")
    print("1. Check if the model version is compatible with your transformers version")
    print("2. Try using the exact same environment as the original TimeSformer paper")
    print("3. Check if there are any preprocessing requirements we're missing")

    return 1

if __name__ == "__main__":
    exit(main())
