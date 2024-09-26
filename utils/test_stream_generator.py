# This script that generates synthetic video frames and streams them to your RTMP server using FFmpeg as a subprocess.
# The script creates moving rectangles and circles with dynamic positions and labels to simulate a live video feed.
# It can be used to i.e. debug the loopback if you're not getting an image into the detection.

import cv2
import numpy as np
import subprocess
import sys
import time

def main():
    # Define FFmpeg command to push video to RTMP server
    width, height, fps = 640, 480, 30  # Frame dimensions and rate
    rtmp_url = 'rtmp://127.0.0.1:1935/live/stream'  # RTMP server URL

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-f', 'rawvideo',  # Input format
        '-vcodec', 'rawvideo',  # Input codec
        '-pix_fmt', 'bgr24',  # Input pixel format
        '-s', f'{width}x{height}',  # Input frame size
        '-r', str(fps),  # Input frame rate
        '-i', '-',  # Input comes from stdin
        '-c:v', 'libx264',  # Output codec
        '-pix_fmt', 'yuv420p',  # Output pixel format
        '-preset', 'veryfast',  # Encoding speed
        '-f', 'flv',  # Output format
        rtmp_url  # Output URL
    ]

    try:
        # Start FFmpeg subprocess
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        print(f"Streaming to {rtmp_url}... Press Ctrl+C to stop.")

        frame_count = 0
        while True:
            # Create a blank frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Calculate dynamic positions for shapes
            rect_x = (frame_count % (width - 100))
            circle_y = (frame_count * 2) % height

            # Draw a moving rectangle
            top_left = (rect_x, 100)
            bottom_right = (rect_x + 100, 200)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle

            # Draw a moving circle
            center = (320, circle_y)
            cv2.circle(frame, center, 50, (255, 0, 0), -1)  # Blue filled circle

            # Add dynamic text
            cv2.putText(frame, f'Frame {frame_count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write frame to FFmpeg's stdin
            try:
                process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("FFmpeg pipe closed. Exiting.")
                break

            frame_count += 1
            time.sleep(1 / fps)  # Maintain the desired frame rate

    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        if process.stdin:
            process.stdin.close()
        process.wait()
        print("FFmpeg subprocess terminated.")

if __name__ == "__main__":
    main()
