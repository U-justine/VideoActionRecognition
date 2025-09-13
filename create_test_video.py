#!/usr/bin/env python3
"""
Create a synthetic test video for verifying the tensor creation fix.
This script generates a simple MP4 video with moving shapes that can be used
to test the video action recognition pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def create_test_video(output_path: Path, duration: int = 5, fps: int = 24, width: int = 640, height: int = 480):
    """
    Create a synthetic test video with moving objects.

    Args:
        output_path: Path where to save the video
        duration: Video duration in seconds
        fps: Frames per second
        width: Video width in pixels
        height: Video height in pixels
    """

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    total_frames = duration * fps
    print(f"Creating video with {total_frames} frames at {fps} FPS...")

    for frame_num in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate animation parameters
        progress = frame_num / total_frames

        # Moving rectangle (simulates "sliding" action)
        rect_x = int(50 + (width - 150) * progress)
        rect_y = height // 2 - 25
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 100, rect_y + 50), (0, 255, 0), -1)

        # Bouncing circle (simulates "bouncing ball" action)
        circle_x = width // 4
        circle_y = int(height // 2 + 100 * np.sin(progress * 4 * np.pi))
        cv2.circle(frame, (circle_x, circle_y), 30, (255, 100, 100), -1)

        # Rotating line (simulates "waving" or "gesturing" action)
        center_x, center_y = 3 * width // 4, height // 2
        angle = progress * 4 * np.pi
        end_x = int(center_x + 80 * np.cos(angle))
        end_y = int(center_y + 80 * np.sin(angle))
        cv2.line(frame, (center_x, center_y), (end_x, end_y), (100, 100, 255), 8)

        # Add frame number for debugging
        cv2.putText(frame, f'Frame {frame_num+1}/{total_frames}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add title
        cv2.putText(frame, 'Test Video - Multiple Actions',
                   (width//2 - 150, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Write frame to video
        out.write(frame)

        if frame_num % 24 == 0:  # Progress update every second
            print(f"  Progress: {frame_num+1}/{total_frames} frames ({(frame_num+1)/total_frames*100:.1f}%)")

    # Clean up
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ Video created successfully: {output_path}")
    print(f"   Duration: {duration} seconds")
    print(f"   Resolution: {width}x{height}")
    print(f"   Frame rate: {fps} FPS")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def create_multiple_test_videos(output_dir: Path):
    """Create several test videos with different characteristics."""

    output_dir.mkdir(exist_ok=True)

    test_configs = [
        {
            "name": "short_action.mp4",
            "duration": 3,
            "fps": 30,
            "width": 640,
            "height": 480,
            "description": "Short 3-second video with basic actions"
        },
        {
            "name": "standard_action.mp4",
            "duration": 5,
            "fps": 24,
            "width": 640,
            "height": 480,
            "description": "Standard 5-second video"
        },
        {
            "name": "hd_action.mp4",
            "duration": 4,
            "fps": 30,
            "width": 1280,
            "height": 720,
            "description": "HD resolution test video"
        },
        {
            "name": "long_action.mp4",
            "duration": 10,
            "fps": 24,
            "width": 640,
            "height": 480,
            "description": "Longer video for extended testing"
        }
    ]

    print("Creating multiple test videos...")
    print("=" * 50)

    for config in test_configs:
        print(f"\n📽️  Creating: {config['name']}")
        print(f"   {config['description']}")

        video_path = output_dir / config['name']
        create_test_video(
            output_path=video_path,
            duration=config['duration'],
            fps=config['fps'],
            width=config['width'],
            height=config['height']
        )

    print(f"\n🎉 All test videos created in: {output_dir}")
    print("\nYou can now use these videos to test the action recognition system:")
    for config in test_configs:
        print(f"  - {config['name']}: {config['description']}")


def main():
    parser = argparse.ArgumentParser(description="Create synthetic test videos for action recognition")
    parser.add_argument("--output", "-o", type=Path, default=Path("test_videos"),
                       help="Output directory for test videos")
    parser.add_argument("--single", "-s", type=str, help="Create single video with this filename")
    parser.add_argument("--duration", "-d", type=int, default=5, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--width", "-w", type=int, default=640, help="Video width")
    parser.add_argument("--height", "-h", type=int, default=480, help="Video height")

    args = parser.parse_args()

    try:
        if args.single:
            # Create single video
            output_path = args.output / args.single
            output_path.parent.mkdir(parents=True, exist_ok=True)

            create_test_video(
                output_path=output_path,
                duration=args.duration,
                fps=args.fps,
                width=args.width,
                height=args.height
            )
        else:
            # Create multiple test videos
            create_multiple_test_videos(args.output)

    except Exception as e:
        print(f"❌ Error creating test video(s): {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
