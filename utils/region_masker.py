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

# If the captured frame is bigger than this, we'll scale it down for display only
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

# Each element: {"name": str, "confidence_threshold": float, "x1": int, "y1": int, "x2": int, "y2": int}
zones = []

# We’ll keep track of the original vs. displayed size so we can transform coordinates back/forth.
original_width = None
original_height = None
display_width = None
display_height = None

drawing = False
ix, iy = -1, -1  # mouse-down coordinates (in *display* coords)
temp_rect = None  # rectangle while dragging (in *display* coords)

# ------------------------------------------------------------------------------
# Logger setup
# ------------------------------------------------------------------------------
logger = logging.getLogger("region_masker")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

# ------------------------------------------------------------------------------
# Load config.ini ([region_masker] section)
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

    video_source = config.get("region_masker", "video_source", fallback=DEFAULT_VIDEO_SOURCE)
    output_json = config.get("region_masker", "output_json", fallback=DEFAULT_OUTPUT_JSON)

    logger.info(f"Using video_source: {video_source}")
    logger.info(f"Using output_json: {output_json}")
    return video_source, output_json

# ------------------------------------------------------------------------------
# Scale the frame if it's larger than our max display size
# Returns the scaled frame, plus the new width/height
# ------------------------------------------------------------------------------
def scale_frame_if_needed(frame):
    global original_width, original_height, display_width, display_height

    original_height, original_width = frame.shape[:2]

    # By default, display_width/height = original
    display_width = original_width
    display_height = original_height

    if display_width > MAX_DISPLAY_WIDTH or display_height > MAX_DISPLAY_HEIGHT:
        # compute scale factors
        scale_w = MAX_DISPLAY_WIDTH / float(display_width)
        scale_h = MAX_DISPLAY_HEIGHT / float(display_height)
        scale = min(scale_w, scale_h)
        new_w = int(display_width * scale)
        new_h = int(display_height * scale)

        logger.info(f"Resizing display from {display_width}x{display_height} to {new_w}x{new_h}")
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        display_width, display_height = new_w, new_h

    return frame

# ------------------------------------------------------------------------------
# Convert display coords -> original coords
# ------------------------------------------------------------------------------
def to_original_coords(x_disp, y_disp):
    """
    Given a coordinate (x_disp, y_disp) in the *display* (possibly scaled),
    return the corresponding coordinate in the *original* frame dimension.
    """
    if display_width == original_width and display_height == original_height:
        # No scaling was done, so they are the same
        return x_disp, y_disp

    # Otherwise, we figure out the scaling ratio
    scale_x = original_width / float(display_width)
    scale_y = original_height / float(display_height)
    x_orig = int(x_disp * scale_x)
    y_orig = int(y_disp * scale_y)
    return x_orig, y_orig

# ------------------------------------------------------------------------------
# Convert original coords -> display coords
# ------------------------------------------------------------------------------
def to_display_coords(x_orig, y_orig):
    """
    Opposite of above: given a coordinate in the original dimension,
    return its coordinate in the display dimension.
    """
    if display_width == original_width and display_height == original_height:
        return x_orig, y_orig

    scale_x = display_width / float(original_width)
    scale_y = display_height / float(original_height)
    x_disp = int(x_orig * scale_x)
    y_disp = int(y_orig * scale_y)
    return x_disp, y_disp

# ------------------------------------------------------------------------------
# Mouse callback for drawing rectangles
# ------------------------------------------------------------------------------
def draw_rectangles(event, x_disp, y_disp, flags, param):
    global ix, iy, drawing, temp_rect

    # (x_disp, y_disp) are the coordinates in the *display* image
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x_disp, y_disp
        temp_rect = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Update the green "live" rectangle (in display coords)
        temp_rect = (ix, iy, x_disp, y_disp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if temp_rect:
            x1_disp, y1_disp, x2_disp, y2_disp = temp_rect

            # Normalize *display* coords
            x1_disp, x2_disp = sorted([x1_disp, x2_disp])
            y1_disp, y2_disp = sorted([y1_disp, y2_disp])

            # Convert them back to *original* coords before storing
            x1_orig, y1_orig = to_original_coords(x1_disp, y1_disp)
            x2_orig, y2_orig = to_original_coords(x2_disp, y2_disp)

            # --- PROMPT user in terminal for zone name ---
            default_name = f"Zone_{len(zones) + 1}"
            zone_name = input(
                f"Enter zone name (press Enter for '{default_name}'): "
            ).strip()
            if not zone_name:
                zone_name = default_name

            # --- PROMPT user for confidence threshold ---
            while True:
                confidence_str = input(
                    "Enter min confidence threshold (0.0–1.0, press Enter = 0.0): "
                ).strip()
                if not confidence_str:  # user just pressed Enter
                    zone_conf = 0.0
                    break
                try:
                    zone_conf = float(confidence_str)
                    if 0.0 <= zone_conf <= 1.0:
                        break
                    else:
                        print("Please enter a number between 0.0 and 1.0.")
                except ValueError:
                    print("Invalid input, please enter a numeric value.")

            zones.append({
                "name": zone_name,
                "confidence_threshold": zone_conf,
                # Store in *original* coords:
                "x1": x1_orig,
                "y1": y1_orig,
                "x2": x2_orig,
                "y2": y2_orig
            })

            logger.info(
                f"Added zone '{zone_name}' w/conf={zone_conf}, "
                f"orig coords=({x1_orig},{y1_orig}) - ({x2_orig},{y2_orig}), "
                f"display coords=({x1_disp},{y1_disp}) - ({x2_disp},{y2_disp})"
            )

        temp_rect = None

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    video_source, output_json = load_config()

    # --------------------------------------------------------------------------
    # Check if there's an existing JSON file with ignore_zones
    # --------------------------------------------------------------------------
    if os.path.exists(output_json):
        logger.info(f"Found existing JSON file: {output_json}, attempting to load.")
        try:
            with open(output_json, 'r') as f:
                data = json.load(f)
            loaded_zones = data.get("ignore_zones", [])
            zones.extend(loaded_zones)  # Add them to our global 'zones' list
            logger.info(f"Loaded {len(loaded_zones)} zones from {output_json}.")
            for i, z in enumerate(loaded_zones, 1):
                name = z.get("name", f"Zone_{i}")
                cthr = z.get("confidence_threshold", 0.0)  # fallback if missing
                x1 = z.get("x1", 0)
                y1 = z.get("y1", 0)
                x2 = z.get("x2", 0)
                y2 = z.get("y2", 0)
                logger.info(
                    f"  -> Zone {i}: name='{name}', threshold={cthr}, coords=({x1},{y1}) - ({x2},{y2})"
                )
        except Exception as e:
            logger.error(f"Failed to parse JSON from {output_json}: {e}")
            logger.info("Continuing with an empty zones list.")
    else:
        logger.info(f"No existing JSON file found at '{output_json}'. Starting with an empty zones list.")

    # --------------------------------------------------------------------------
    # Now open the stream and proceed
    # --------------------------------------------------------------------------
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {video_source}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.error("Could not read a frame from the stream.")
        return

    # Possibly scale down the frame for display
    frame_display = scale_frame_if_needed(frame)

    cv2.namedWindow("Draw Ignore Zones", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw Ignore Zones", draw_rectangles)

    while True:
        display_frame = frame_display.copy()

        # Draw existing zones in red (convert their *original* coords to display coords)
        for z in zones:
            x1_orig, y1_orig = z.get("x1", 0), z.get("y1", 0)
            x2_orig, y2_orig = z.get("x2", 0), z.get("y2", 0)

            # Convert original -> display
            x1_disp, y1_disp = to_display_coords(x1_orig, y1_orig)
            x2_disp, y2_disp = to_display_coords(x2_orig, y2_orig)

            cv2.rectangle(display_frame, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 0, 255), 2)

            name = z.get('name', 'Zone_?')
            cthr = z.get('confidence_threshold', 0.0)
            label = f"{name} ({cthr})"
            # Place the label above or below this box as needed
            label_y = y1_disp - 5 if (y1_disp - 10) > 0 else y1_disp + 15
            cv2.putText(
                display_frame,
                label,
                (x1_disp, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Draw the in-progress rectangle in green (temp_rect is in *display* coords)
        if temp_rect:
            x1_disp, y1_disp, x2_disp, y2_disp = temp_rect
            cv2.rectangle(display_frame, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 255, 0), 2)

        # Instructions
        cv2.putText(display_frame,
                    "Left-click & drag to draw. Press 's' to save, 'q' to quit.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2)

        cv2.imshow("Draw Ignore Zones", display_frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            # Save to JSON (all zones in *original* coords)
            if os.path.exists(output_json):
                logger.info(f"Overwriting existing file: {output_json}")
            with open(output_json, 'w') as f:
                json.dump({"ignore_zones": zones}, f, indent=4)
            logger.info(f"Zones saved to {output_json}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
