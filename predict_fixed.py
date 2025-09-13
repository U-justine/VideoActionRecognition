#!/usr/bin/env python3
"""
Fixed video action prediction with proper TimeSformer tensor format.
This version resolves the tensor compatibility issues definitively.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Video reading libraries
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

try:
    import decord
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    decord = None

MODEL_ID = "facebook/timesformer-base-finetuned-k400"

def read_video_frames_cv2(video_path: Path, num_frames: int = 8) -> List[Image.Image]:
    """Read frames using OpenCV with robust error handling."""
    if not HAS_CV2:
        raise RuntimeError("OpenCV not available")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise RuntimeError("Video has no frames")

    # Sample frames uniformly across the video
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        step = max(1, total_frames // num_frames)
        frame_indices = [i * step for i in range(num_frames)]
        # Ensure we don't exceed total frames
        frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

    cap.release()

    # Pad with last frame if needed
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            # Create black frame as fallback
            black_frame = Image.new('RGB', (224, 224), (0, 0, 0))
            frames.append(black_frame)

    return frames[:num_frames]

def read_video_frames_decord(video_path: Path, num_frames: int = 8) -> List[Image.Image]:
    """Read frames using decord."""
    if not HAS_DECORD:
        raise RuntimeError("Decord not available")

    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)

    if total_frames == 0:
        raise RuntimeError("Video has no frames")

    # Sample frames
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        step = max(1, total_frames // num_frames)
        indices = [i * step for i in range(num_frames)]
        indices = [min(idx, total_frames - 1) for idx in indices]

    try:
        frame_arrays = vr.get_batch(indices).asnumpy()
        frames = [Image.fromarray(frame) for frame in frame_arrays]
    except Exception:
        # Fallback to individual frame reading
        frames = []
        for idx in indices:
            try:
                frame = vr[idx].asnumpy()
                frames.append(Image.fromarray(frame))
            except Exception:
                continue

    # Pad if necessary
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            black_frame = Image.new('RGB', (224, 224), (0, 0, 0))
            frames.append(black_frame)

    return frames[:num_frames]

def read_video_frames(video_path: Path, num_frames: int = 8) -> List[Image.Image]:
    """Read video frames with fallback methods."""
    last_error = None

    # Try decord first (usually faster and more reliable)
    if HAS_DECORD:
        try:
            frames = read_video_frames_decord(video_path, num_frames)
            if frames and len(frames) > 0:
                logging.debug(f"Successfully read {len(frames)} frames using decord")
                return frames
        except Exception as e:
            last_error = e
            logging.debug(f"Decord failed: {e}")

    # Fallback to OpenCV
    if HAS_CV2:
        try:
            frames = read_video_frames_cv2(video_path, num_frames)
            if frames and len(frames) > 0:
                logging.debug(f"Successfully read {len(frames)} frames using OpenCV")
                return frames
        except Exception as e:
            last_error = e
            logging.debug(f"OpenCV failed: {e}")

    if last_error:
        raise RuntimeError(f"Failed to read video frames: {last_error}")
    else:
        raise RuntimeError("No video reading library available")

def normalize_frames(frames: List[Image.Image], target_size: Tuple[int, int] = (224, 224)) -> List[Image.Image]:
    """Normalize frames to consistent format."""
    if not frames:
        raise RuntimeError("No frames to normalize")

    normalized = []
    for i, frame in enumerate(frames):
        try:
            # Convert to RGB if needed
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')

            # Resize to target size
            if frame.size != target_size:
                frame = frame.resize(target_size, Image.Resampling.LANCZOS)

            normalized.append(frame)
        except Exception as e:
            logging.warning(f"Error normalizing frame {i}: {e}")
            # Create a black frame as fallback
            black_frame = Image.new('RGB', target_size, (0, 0, 0))
            normalized.append(black_frame)

    return normalized

def create_timesformer_tensor(frames: List[Image.Image]) -> torch.Tensor:
    """
    Create properly formatted tensor for TimeSformer model.

    TimeSformer expects 5D input tensor:
    Input format: [batch_size, num_frames, channels, height, width]
    For 8 frames of 224x224: [1, 8, 3, 224, 224]
    """
    if len(frames) != 8:
        raise ValueError(f"Expected 8 frames, got {len(frames)}")

    # Convert frames to tensors without using numpy
    frame_tensors = []

    for frame in frames:
        # Ensure correct format
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')
        if frame.size != (224, 224):
            frame = frame.resize((224, 224), Image.Resampling.LANCZOS)

        # Convert PIL image to tensor manually to avoid numpy issues
        pixels = list(frame.getdata())  # List of (R, G, B) tuples

        # Separate into RGB channels and normalize
        r_channel = []
        g_channel = []
        b_channel = []

        for r, g, b in pixels:
            r_channel.append(r / 255.0)
            g_channel.append(g / 255.0)
            b_channel.append(b / 255.0)

        # Reshape to 2D (224, 224) for each channel
        r_tensor = torch.tensor(r_channel, dtype=torch.float32).view(224, 224)
        g_tensor = torch.tensor(g_channel, dtype=torch.float32).view(224, 224)
        b_tensor = torch.tensor(b_channel, dtype=torch.float32).view(224, 224)

        # Stack channels: (3, 224, 224)
        frame_tensor = torch.stack([r_tensor, g_tensor, b_tensor], dim=0)
        frame_tensors.append(frame_tensor)

    # Stack frames: (8, 3, 224, 224)
    video_tensor = torch.stack(frame_tensors, dim=0)

    # Rearrange to TimeSformer format: (batch, frames, channels, height, width)
    # From (8, 3, 224, 224) to (1, 8, 3, 224, 224)
    video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension: (1, 8, 3, 224, 224)

    logging.debug(f"Created tensor with shape: {video_tensor.shape}")
    logging.debug(f"Tensor dtype: {video_tensor.dtype}")
    logging.debug(f"Tensor range: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")

    return video_tensor

def load_model(device: Optional[str] = None):
    """Load TimeSformer model and processor."""
    try:
        from transformers import AutoImageProcessor, TimesformerForVideoClassification

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading model on device: {device}")

        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = TimesformerForVideoClassification.from_pretrained(MODEL_ID)
        model.to(device)
        model.eval()

        logging.info("Model loaded successfully")
        return processor, model, device

    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

def predict_actions(video_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Predict actions in video using TimeSformer model.

    Args:
        video_path: Path to video file
        top_k: Number of top predictions to return

    Returns:
        List of (action_label, confidence_score) tuples
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        # Load model
        processor, model, device = load_model()

        # Extract and normalize frames
        logging.info(f"Processing video: {video_path.name}")
        frames = read_video_frames(video_path, num_frames=8)
        frames = normalize_frames(frames, target_size=(224, 224))

        logging.info(f"Extracted and normalized {len(frames)} frames")

        # Create tensor in correct format
        pixel_values = create_timesformer_tensor(frames)
        pixel_values = pixel_values.to(device)

        # Run inference
        logging.info("Running model inference...")
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

        # Get top-k predictions
        probabilities = torch.softmax(logits, dim=-1)[0]  # Remove batch dimension
        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        # Convert to results
        results = []
        for prob, idx in zip(top_probs, top_indices):
            label = model.config.id2label[idx.item()]
            confidence = float(prob.item())
            results.append((label, confidence))

        logging.info(f"Generated {len(results)} predictions successfully")

        # Log top prediction for debugging
        if results:
            top_label, top_conf = results[0]
            logging.info(f"Top prediction: {top_label} ({top_conf:.3f})")

        return results

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise RuntimeError(f"Video processing error: {e}")

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Predict actions in video using TimeSformer")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Run prediction
        predictions = predict_actions(args.video, top_k=args.top_k)

        if args.json:
            output = [{"label": label, "confidence": confidence}
                     for label, confidence in predictions]
            print(json.dumps(output, indent=2))
        else:
            print(f"\nTop {len(predictions)} predictions for: {args.video}")
            print("-" * 60)
            for i, (label, confidence) in enumerate(predictions, 1):
                print(f"{i:2d}. {label:<35} {confidence:.4f}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
