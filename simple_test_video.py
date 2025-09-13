#!/usr/bin/env python3
"""
Simple test video creator for TimeSformer testing.
Creates a basic MP4 video with simple motion patterns.
"""

import cv2
import numpy as np
from pathlib import Path

def create_simple_test_video(output_path: str = "test_video.mp4", duration_seconds: int = 3):
    """Create a simple test video with moving shapes."""

    # Video properties
    width, height = 320, 240
    fps = 30
    total_frames = duration_seconds * fps

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating test video: {output_path}")
    print(f"Duration: {duration_seconds} seconds, {total_frames} frames")

    for frame_num in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add background gradient
        for y in range(height):
            for x in range(width):
                frame[y, x] = [
                    int(255 * (x / width)),  # Red gradient
                    int(255 * (y / height)),  # Green gradient
                    128  # Blue constant
                ]

        # Add moving circle (simulates motion)
        progress = frame_num / total_frames
        center_x = int(50 + (width - 100) * progress)
        center_y = int(height // 2 + 30 * np.sin(progress * 4 * np.pi))
        radius = 20 + int(10 * np.sin(progress * 6 * np.pi))

        cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)

        # Add moving rectangle (more motion)
        rect_x = int(width - 80 - (width - 160) * progress)
        rect_y = int(20 + 20 * np.cos(progress * 3 * np.pi))
        cv2.rectangle(frame,
                     (rect_x, rect_y),
                     (rect_x + 40, rect_y + 30),
                     (0, 255, 255), -1)

        # Add frame counter for debugging
        cv2.putText(frame, f"Frame {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"✅ Video created successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    output_file = "test_video.mp4"
    create_simple_test_video(output_file, duration_seconds=5)

    # Verify the file was created
    if Path(output_file).exists():
        file_size = Path(output_file).stat().st_size
        print(f"File size: {file_size / 1024:.1f} KB")
    else:
        print("❌ Failed to create video file")
