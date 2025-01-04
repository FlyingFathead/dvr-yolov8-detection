#!/usr/bin/env python3
# File: ./utils/region_masker.py

import cv2
import json
import os
import sys
import configparser
import logging

# --------------------------
# Default fallback settings:
# --------------------------
DEFAULT_VIDEO_SOURCE = "rtmp://127.0.0.1:1935/live/stream"
DEFAULT_OUTPUT_JSON = "ignore_zones.json"

# Data structure to hold the rectangles
zones = []  # Each element: {"name": str, "x1": int, "y1": int, "x2": int, "y2": int}

# For drawing the rectangle on screen
drawing = False
ix, iy = -1, -1  # initial x,y when mouse is pressed
temp_rect = None  # temp rectangle while dragging

# ------------------------------------------------------------------------------
# Quick logger setup (or integrate with your main logging if you prefer)
# ------------------------------------------------------------------------------
logger = logging.getLogger("region_masker")
logger.setLevel(logging.INFO)
# Print logs to stdout
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

# ------------------------------------------------------------------------------
# Load config.ini (if present), read [region_masker] section
# ------------------------------------------------------------------------------
def load_config():
    """Load 'video_source' and 'output_json' from config.ini if present,
       otherwise fallback to defaults and log warnings."""
    config = configparser.ConfigParser()
    read_files = config.read("config.ini")

    if not read_files:
        logger.warning("No config.ini found or it's empty. Using default settings.")
        return DEFAULT_VIDEO_SOURCE, DEFAULT_OUTPUT_JSON

    if not config.has_section("region_masker"):
        logger.warning("[region_masker] section not found in config.ini. Using defaults.")
        return DEFAULT_VIDEO_SOURCE, DEFAULT_OUTPUT_JSON

    # Attempt to read from [region_masker] section with fallback to defaults
    video_source = config.get("region_masker", "video_source", fallback=DEFAULT_VIDEO_SOURCE)
    output_json = config.get("region_masker", "output_json", fallback=DEFAULT_OUTPUT_JSON)

    logger.info(f"Using video_source: {video_source}")
    logger.info(f"Using output_json: {output_json}")
    return video_source, output_json

# ------------------------------------------------------------------------------
# Mouse callback for drawing rectangles
# ------------------------------------------------------------------------------
def draw_rectangles(event, x, y, flags, param):
    global ix, iy, drawing, temp_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing
        drawing = True
        ix, iy = x, y
        temp_rect = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the rectangle as we drag
            temp_rect = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing
        drawing = False
        if temp_rect:
            x1, y1, x2, y2 = temp_rect
            # Normalize coordinates so x1 < x2, y1 < y2
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            # Possibly ask user for a name or use a default
            rect_name = f"Zone_{len(zones)+1}"
            zones.append({
                "name": rect_name,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2
            })
        temp_rect = None

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    # 1) Load config (or defaults)
    video_source, output_json = load_config()

    # 2) Open the stream
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {video_source}")
        return

    # 3) Grab just 1 frame (or you could do a loop if you want more frames)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.error("Could not read a frame from the stream.")
        return

    # 4) Setup the window and callback
    cv2.namedWindow("Draw Ignore Zones")
    cv2.setMouseCallback("Draw Ignore Zones", draw_rectangles)

    # 5) Main loop to display and wait for user input
    while True:
        display_frame = frame.copy()

        # Draw the rectangles in 'zones'
        for z in zones:
            cv2.rectangle(display_frame,
                          (z["x1"], z["y1"]),
                          (z["x2"], z["y2"]),
                          (0, 0, 255), 2)  # Red boxes

        # If currently drawing a rect, show the in-progress one in green
        if temp_rect:
            x1, y1, x2, y2 = temp_rect
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display instructions
        cv2.putText(display_frame,
                    "Left-drag to draw. Press 's' to save, 'q' to quit.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2)

        cv2.imshow("Draw Ignore Zones", display_frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            # Save the zones to JSON
            if os.path.exists(output_json):
                logger.info(f"Overwriting existing file: {output_json}")
            with open(output_json, 'w') as f:
                json.dump({"ignore_zones": zones}, f, indent=4)
            logger.info(f"Zones saved to {output_json}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
