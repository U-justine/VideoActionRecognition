#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import decord  # type: ignore
    _decord_error = None
except Exception as e:  # pragma: no cover
    _decord_error = e
    decord = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification

MODEL_ID = "facebook/timesformer-base-finetuned-k400"

def fix_numpy_compatibility():
    """Check and fix NumPy compatibility issues."""
    try:
        # Test basic numpy operations that are used in video processing
        test_array = np.array([1, 2, 3], dtype=np.float32)
        # Test stacking operations
        np.stack([test_array, test_array])

        # Test array creation and manipulation
        test_image_array = np.zeros((224, 224, 3), dtype=np.float32)
        test_video_array = np.stack([test_image_array, test_image_array], axis=0)

        # If we reach here, numpy is working
        logging.debug(f"NumPy {np.__version__} compatibility check passed")
        return True

    except Exception as e:
        logging.warning(f"NumPy compatibility issue: {e}")

        # For NumPy 2.x compatibility, try alternative approaches
        try:
            # Alternative stack operation that works with both versions
            test_list = [test_array, test_array]
            stacked = np.array(test_list)
            logging.info("Using NumPy 2.x compatible operations")
            return True
        except Exception as e2:
            logging.error(f"NumPy compatibility cannot be resolved: {e2}")
            return False

def _read_video_frames_decord(video_path: Path, num_frames: int) -> List[Image.Image]:
    """Read video frames using decord library."""
    vr = decord.VideoReader(str(video_path))
    total = len(vr)

    if total == 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    # Handle edge case where video has fewer frames than requested
    actual_num_frames = min(num_frames, total)
    if actual_num_frames <= 0:
        raise RuntimeError(f"Invalid frame count: {actual_num_frames}")

    indices = np.linspace(0, total - 1, num=actual_num_frames, dtype=int).tolist()

    try:
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames]
    except Exception as e:
        logging.warning(f"Decord batch read failed: {e}")
        # Fallback to individual frame reading
        frames = []
        for idx in indices:
            try:
                frame = vr[idx].asnumpy()
                frames.append(Image.fromarray(frame))
            except Exception:
                continue
        return frames

def _read_video_frames_cv2(video_path: Path, num_frames: int) -> List[Image.Image]:
    """Read video frames using OpenCV."""
    if cv2 is None:
        raise RuntimeError("OpenCV (opencv-python) is required if decord is not installed.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    # Handle edge case where video has fewer frames than requested
    actual_num_frames = min(num_frames, total)
    if actual_num_frames <= 0:
        raise RuntimeError(f"Invalid frame count: {actual_num_frames}")

    indices = np.linspace(0, max(total - 1, 0), num=actual_num_frames, dtype=int).tolist()

    result: List[Image.Image] = []
    current_idx = 0
    frame_pos_set_ok = hasattr(cv2, "CAP_PROP_POS_FRAMES")

    for target in indices:
        try:
            if frame_pos_set_ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
                ok, frame = cap.read()
                if not ok:
                    continue
            else:
                # Fallback: read sequentially until we reach target
                while current_idx <= target:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    current_idx += 1
                if not ok:
                    continue

            # Convert BGR->RGB and to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result.append(Image.fromarray(frame_rgb))
        except Exception as e:
            logging.warning(f"Error reading frame {target}: {e}")
            continue

    cap.release()
    return result

def _read_video_frames(video_path: Path, num_frames: int) -> List[Image.Image]:
    """Read uniformly sampled frames using decord if available, otherwise OpenCV."""
    frames = []
    last_error = None

    # Try decord first
    if decord is not None:
        try:
            frames = _read_video_frames_decord(video_path, num_frames)
            if frames:
                logging.debug(f"Successfully read {len(frames)} frames using decord")
                return frames
        except Exception as e:
            last_error = e
            logging.warning(f"Decord failed: {e}")

    # Fallback to OpenCV
    try:
        frames = _read_video_frames_cv2(video_path, num_frames)
        if frames:
            logging.debug(f"Successfully read {len(frames)} frames using OpenCV")
            return frames
    except Exception as e:
        last_error = e
        logging.warning(f"OpenCV failed: {e}")

    # If both failed, raise the last error
    if last_error:
        raise RuntimeError(f"Failed to read video frames: {last_error}")
    else:
        raise RuntimeError("No video reading library available")

def normalize_frames(frames: List[Image.Image], required_frames: int, target_size: Tuple[int, int] = (224, 224)) -> List[Image.Image]:
    """Normalize frames to required count and size."""
    if not frames:
        raise RuntimeError("No frames to normalize")

    # Adjust frame count
    original_count = len(frames)
    if len(frames) < required_frames:
        # Pad by repeating frames cyclically
        padding_needed = required_frames - len(frames)
        for i in range(padding_needed):
            frames.append(frames[i % original_count])
        logging.info(f"Padded frames from {original_count} to {required_frames}")
    elif len(frames) > required_frames:
        # Uniformly sample frames
        indices = np.linspace(0, len(frames) - 1, num=required_frames, dtype=int)
        frames = [frames[i] for i in indices]
        logging.info(f"Sampled {required_frames} frames from {original_count}")

    # Normalize frame properties
    normalized_frames = []
    for i, frame in enumerate(frames):
        try:
            # Ensure RGB mode
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')

            # Resize to target size
            if frame.size != target_size:
                frame = frame.resize(target_size, Image.Resampling.LANCZOS)

            normalized_frames.append(frame)
        except Exception as e:
            logging.error(f"Error normalizing frame {i}: {e}")
            # Create a black frame as fallback
            black_frame = Image.new('RGB', target_size, (0, 0, 0))
            normalized_frames.append(black_frame)

    return normalized_frames

def create_tensor_from_frames(frames: List[Image.Image], processor=None) -> torch.Tensor:
    """Create tensor from frames using multiple fallback strategies."""

    # Strategy 1: Use processor if available and working
    if processor is not None:
        strategies = [
            lambda: processor(images=frames, return_tensors="pt"),
            lambda: processor(videos=frames, return_tensors="pt"),
            lambda: processor(frames, return_tensors="pt"),
        ]

        for i, strategy in enumerate(strategies, 1):
            try:
                inputs = strategy()
                if 'pixel_values' in inputs:
                    tensor = inputs['pixel_values']
                    logging.info(f"Strategy {i} succeeded, tensor shape: {tensor.shape}")
                    return tensor
            except Exception as e:
                logging.debug(f"Processor strategy {i} failed: {e}")
                continue

    # Strategy 2: Direct PyTorch tensor creation (bypass numpy compatibility issues)
    try:
        logging.info("Using direct PyTorch tensor creation")

        # Convert frames directly to PyTorch tensors
        frame_tensors = []
        for i, frame in enumerate(frames):
            # Ensure frame is in the right format
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            if frame.size != (224, 224):
                frame = frame.resize((224, 224), Image.Resampling.LANCZOS)

            # Get pixel data and reshape properly
            pixels = list(frame.getdata())
            logging.debug(f"Frame {i}: got {len(pixels)} pixels")

            # Create tensor with shape (height, width, channels)
            pixel_tensor = torch.tensor(pixels, dtype=torch.float32).view(224, 224, 3)
            pixel_tensor = pixel_tensor / 255.0  # Normalize to [0, 1]
            logging.debug(f"Frame {i} tensor shape: {pixel_tensor.shape}")
            frame_tensors.append(pixel_tensor)

        # Stack frames into video tensor: (num_frames, height, width, channels)
        video_tensor = torch.stack(frame_tensors, dim=0)
        logging.debug(f"Stacked tensor shape: {video_tensor.shape}")

        # Rearrange dimensions for TimeSformer: (batch, channels, num_frames, height, width)
        # Current: (num_frames=8, height=224, width=224, channels=3)
        # Target:  (batch=1, num_frames=8, channels=3, height=224, width=224)
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (frames, height, width, channels) -> (frames, channels, height, width)
        logging.debug(f"After first permute: {video_tensor.shape}")

        video_tensor = video_tensor.unsqueeze(0)  # (frames, channels, height, width) -> (1, frames, channels, height, width)
        logging.debug(f"After second permute and unsqueeze: {video_tensor.shape}")

        logging.info(f"Direct tensor creation succeeded, final shape: {video_tensor.shape}")
        return video_tensor

    except Exception as e:
        logging.debug(f"Direct tensor creation failed: {e}")

    # Strategy 3: Manual tensor creation with numpy fallback
    try:
        logging.info("Using numpy-based tensor creation")

        # Convert frames to numpy arrays
        frame_arrays = []
        for frame in frames:
            # Ensure frame is in the right format
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            if frame.size != (224, 224):
                frame = frame.resize((224, 224), Image.Resampling.LANCZOS)

            # Convert to array and normalize
            frame_array = np.array(frame, dtype=np.float32)
            frame_array = frame_array / 255.0  # Normalize to [0, 1]
            frame_arrays.append(frame_array)

        # Stack frames: (num_frames, height, width, channels)
        try:
            video_array = np.stack(frame_arrays, axis=0)
        except Exception:
            # Fallback for compatibility issues
            video_array = np.array(frame_arrays)

        # Convert to PyTorch tensor
        video_tensor = torch.from_numpy(video_array)
        logging.debug(f"Numpy tensor initial shape: {video_tensor.shape}")

        # Rearrange dimensions for TimeSformer: (batch, num_frames, channels, height, width)
        # Current: (num_frames, height, width, channels)
        # Target:  (batch, num_frames, channels, height, width)
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (frames, height, width, channels) -> (frames, channels, height, width)
        video_tensor = video_tensor.unsqueeze(0)  # (frames, channels, height, width) -> (1, frames, channels, height, width)

        logging.info(f"Numpy tensor creation succeeded, shape: {video_tensor.shape}")
        return video_tensor

    except Exception as e:
        logging.debug(f"Numpy tensor creation failed: {e}")

    # Strategy 4: Pure Python fallback (slowest but most compatible)
    try:
        logging.info("Using pure Python tensor creation")

        # Convert frames to pure Python lists
        video_data = []
        for frame in frames:
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            if frame.size != (224, 224):
                frame = frame.resize((224, 224), Image.Resampling.LANCZOS)

            # Get pixel data as list of RGB tuples
            pixels = list(frame.getdata())

            # Convert to 3D array structure: [height][width][channels]
            frame_data = []
            for row in range(224):
                row_data = []
                for col in range(224):
                    pixel_idx = row * 224 + col
                    r, g, b = pixels[pixel_idx]
                    row_data.append([r/255.0, g/255.0, b/255.0])  # Normalize
                frame_data.append(row_data)
            video_data.append(frame_data)

        # Convert to tensor
        video_tensor = torch.tensor(video_data, dtype=torch.float32)
        logging.debug(f"Pure Python tensor initial shape: {video_tensor.shape}")

        # Rearrange dimensions: (frames, height, width, channels) -> (batch, frames, channels, height, width)
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (frames, height, width, channels) -> (frames, channels, height, width)
        video_tensor = video_tensor.unsqueeze(0)  # (frames, channels, height, width) -> (1, frames, channels, height, width)

        logging.info(f"Pure Python tensor creation succeeded, shape: {video_tensor.shape}")
        return video_tensor

    except Exception as e:
        raise RuntimeError(f"All tensor creation strategies failed. Last error: {e}")

def load_model(device: Optional[str] = None):
    """Load the TimeSformer model and processor."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        logging.info("Loading TimeSformer model...")
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = TimesformerForVideoClassification.from_pretrained(MODEL_ID)
        model.to(device)
        model.eval()
        logging.info(f"Model loaded successfully on {device}")
        return processor, model, device
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

def predict_actions(video_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Run inference on a video and return top-k (label, score)."""

    # Check numpy compatibility first
    if not fix_numpy_compatibility():
        logging.warning("NumPy compatibility issues detected, but continuing with fallbacks")
        # Don't fail completely - try to continue with available functionality

    try:
        processor, model, device = load_model()
        required_frames = int(getattr(model.config, "num_frames", 8))

        logging.info(f"Processing video: {video_path}")
        logging.info(f"Required frames: {required_frames}")

        # Read video frames
        frames = _read_video_frames(Path(video_path), num_frames=required_frames)
        if not frames:
            raise RuntimeError("Could not extract any frames from the video")

        logging.info(f"Extracted {len(frames)} frames")

        # Normalize frames
        frames = normalize_frames(frames, required_frames)
        logging.info(f"Normalized to {len(frames)} frames")

        # Create tensor
        pixel_values = create_tensor_from_frames(frames, processor)

        # Move to device
        pixel_values = pixel_values.to(device)

        # Run inference
        logging.info("Running inference...")
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)[0]

            # Get top-k predictions
            scores, indices = torch.topk(probs, k=top_k)

            # Convert to labels
            results = []
            for score, idx in zip(scores.cpu(), indices.cpu()):
                label = model.config.id2label[idx.item()]
                results.append((label, float(score)))

            logging.info("Prediction completed successfully")
            return results

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise RuntimeError(f"Video processing error: {e}")

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Predict actions in a video using TimeSformer")
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k predictions to show")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        preds = predict_actions(args.video, top_k=args.top_k)

        if args.json:
            print(json.dumps([{"label": l, "score": s} for l, s in preds], indent=2))
        else:
            print(f"\nTop {len(preds)} predictions for: {args.video}")
            print("-" * 50)
            for i, (label, score) in enumerate(preds, 1):
                print(f"{i:2d}. {label:<30} ({score:.3f})")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
