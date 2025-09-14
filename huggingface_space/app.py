import gradio as gr
import torch
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from typing import List, Tuple
import time
from transformers import TimesformerImageProcessor, TimesformerForVideoClassification
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

# Model configuration
MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
FRAMES_PER_VIDEO = 32

# Global variables for model and processor
model = None
processor = None
device = None
id2label = None

def load_model():
    """Load the TimeSformer model and processor."""
    global model, processor, device, id2label

    print("🔄 Loading TimeSformer model...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {device}")

        # Load processor
        processor = TimesformerImageProcessor.from_pretrained(MODEL_NAME)

        # Load model
        model = TimesformerForVideoClassification.from_pretrained(MODEL_NAME)
        model = model.to(device)
        model.eval()

        # Get label mapping
        id2label = model.config.id2label

        print(f"✅ Model loaded successfully! Can recognize {len(id2label)} actions.")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def extract_frames_cv2(video_path: str, target_frames: int = FRAMES_PER_VIDEO) -> np.ndarray:
    """Extract uniformly sampled frames from video using OpenCV."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"📹 Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")

    # Calculate frame indices to sample
    if total_frames <= target_frames:
        frame_indices = list(range(total_frames))
        frame_indices.extend([total_frames - 1] * (target_frames - total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            if frames:
                frames.append(frames[-1])
            else:
                raise ValueError(f"Cannot read frame {frame_idx}")

    cap.release()
    return np.array(frames)

def predict_video_action(video_path: str, top_k: int = 5) -> Tuple[str, List[Tuple[str, float]]]:
    """Predict actions in a video and return formatted results."""
    if model is None or processor is None:
        return "❌ Model not loaded. Please refresh the page.", []

    try:
        print(f"🎯 Analyzing video: {Path(video_path).name}")

        # Extract frames
        start_time = time.time()
        frames = extract_frames_cv2(video_path)
        extract_time = time.time() - start_time

        # Process frames
        inputs = processor(list(frames), return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)

        # Predict
        with torch.no_grad():
            outputs = model(pixel_values)
            logits = outputs.logits

        # Get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        total_time = time.time() - start_time

        # Get top-k predictions
        top_k_values, top_k_indices = torch.topk(probabilities, top_k, dim=-1)

        predictions = []
        for i in range(top_k):
            idx = top_k_indices[0][i].item()
            confidence = top_k_values[0][i].item()
            action = id2label[idx]
            predictions.append((action, confidence))

        # Format results
        result_text = f"🎬 **Analysis Complete!**\n\n"
        result_text += f"⏱️ Processing time: {total_time:.2f}s\n"
        result_text += f"🎯 Top {top_k} predictions:\n\n"

        for i, (action, confidence) in enumerate(predictions, 1):
            percentage = confidence * 100
            if i == 1:
                result_text += f"🏆 **{i}. {action}** - {percentage:.1f}%\n"
            else:
                result_text += f"{i}. {action} - {percentage:.1f}%\n"

        return result_text, predictions

    except Exception as e:
        error_msg = f"❌ Error analyzing video: {str(e)}\n\n"
        error_msg += "💡 **Tips:**\n"
        error_msg += "- Ensure your video file is not corrupted\n"
        error_msg += "- Try MP4 format for best compatibility\n"
        error_msg += "- Make sure the video contains clear human actions\n"
        error_msg += "- Keep video size under 100MB for faster processing"

        return error_msg, []

def gradio_predict(video_file):
    """Gradio interface function."""
    if video_file is None:
        return "Please upload a video file first.", None

    try:
        # Get predictions
        result_text, predictions = predict_video_action(video_file, top_k=8)

        # Create confidence chart data
        if predictions:
            labels = [pred[0][:30] + "..." if len(pred[0]) > 30 else pred[0] for pred in predictions]
            values = [pred[1] * 100 for pred in predictions]

            # Create a simple bar chart using Gradio's plotting
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(labels[::-1], values[::-1], color=['#667eea', '#764ba2', '#f093fb', '#48bb78', '#ed8936', '#9f7aea', '#38b2ac', '#e53e3e'])

            ax.set_xlabel('Confidence (%)')
            ax.set_title('Top Action Predictions', fontsize=16, fontweight='bold')
            ax.set_xlim(0, 100)

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values[::-1])):
                ax.text(value + 1, bar.get_y() + bar.get_height()/2, f'{value:.1f}%',
                       va='center', fontweight='bold')

            plt.tight_layout()

            # Save plot to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
                plot_path = tmp.name

            plt.close()
            return result_text, plot_path

        return result_text, None

    except Exception as e:
        return f"❌ Unexpected error: {str(e)}", None

# Load model on startup
print("🚀 Initializing Video Action Recognition...")
model_loaded = load_model()

if not model_loaded:
    print("⚠️ Failed to load model. The app may not work correctly.")

# Create Gradio interface
title = "🎬 AI Video Action Recognition"
description = """
Upload a video and get AI-powered predictions of human actions using Facebook's TimeSformer model!

**What it can recognize:**
- Sports activities (basketball, tennis, swimming, etc.)
- Daily activities (cooking, cleaning, reading, etc.)
- Exercise and fitness (yoga, running, weightlifting, etc.)
- Musical performances (dancing, playing instruments, etc.)
- Work activities (typing, writing, presenting, etc.)
- Social interactions (waving, shaking hands, etc.)

**Tips for best results:**
- Use clear, well-lit videos (2-10 seconds work well)
- Ensure the action is clearly visible
- MP4 format recommended
- Keep file size under 100MB
"""

article = """
### About This Model
This application uses Facebook's TimeSformer model fine-tuned on the Kinetics-400 dataset,
which contains 400 different action classes. The model processes 32 uniformly sampled frames
from your video to make predictions.

### Technical Details
- **Model**: facebook/timesformer-base-finetuned-k400
- **Dataset**: Kinetics-400 (400 action classes)
- **Input**: 32 frames per video
- **Processing**: Runs on GPU when available

### Links
- [GitHub Repository](https://github.com/u-justine/VideoActionRecognition)
- [Model on Hugging Face](https://huggingface.co/facebook/timesformer-base-finetuned-k400)
- [Research Paper](https://arxiv.org/abs/2102.05095)

**Built with ❤️ using TimeSformer and Gradio**
"""

# Create the interface
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Video(
        label="Upload Video",
        sources=["upload"],
        include_audio=False,
        max_length=30,  # 30 seconds max
        height=400
    ),
    outputs=[
        gr.Textbox(
            label="Predictions",
            lines=15,
            max_lines=20,
            show_copy_button=True
        ),
        gr.Image(
            label="Confidence Chart",
            type="filepath"
        )
    ],
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
    examples=[
        # You can add example videos here if you have them
        # ["example_video.mp4"],
    ],
    cache_examples=False,
    flagging_mode="never"
)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
