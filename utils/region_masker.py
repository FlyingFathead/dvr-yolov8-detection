#!/usr/bin/env python3
# File: ./utils/region_masker.py

import cv2
import json
import os
import sys
import signal  # <-- We'll catch Ctrl-C
import configparser
import logging
from pathlib import Path

# ---------------------------------------------------------------------
# Default fallback settings
# ---------------------------------------------------------------------
DEFAULT_VIDEO_SOURCE = "rtmp://127.0.0.1:1935/live/stream"
DEFAULT_MASKED_ZONES_JSON = "ignore_zones.json"
DEFAULT_NAMED_ZONES_JSON = "named_zones.json"

MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

zones = []  # We'll store whichever set of zones (masked or named) in a single list
original_width = None
original_height = None
display_width = None
display_height = None

drawing = False
ix, iy = -1, -1
temp_rect = None

zone_mode = None
zone_output_file = None
use_critical_thresholds = False  # If true, prompt for a "critical_threshold" in named mode

logger = logging.getLogger("region_masker")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

def load_config():
    config = configparser.ConfigParser()
    found_files = config.read("config.ini")

    if not found_files:
        logger.warning("No config.ini found or it's empty. Using default settings.")
        return (
            False,
            DEFAULT_VIDEO_SOURCE,
            False,  # enable_masked
            DEFAULT_MASKED_ZONES_JSON,
            False,  # enable_named
            DEFAULT_NAMED_ZONES_JSON,
            False
        )

    logger.info(f"Found config file(s): {found_files}")

    if not config.has_section("region_masker"):
        logger.warning("[region_masker] section not found in config.ini. Using default settings.")
        return (
            True,
            DEFAULT_VIDEO_SOURCE,
            False,
            DEFAULT_MASKED_ZONES_JSON,
            False,
            DEFAULT_NAMED_ZONES_JSON,
            False
        )

    video_source = config.get("region_masker", "video_source", fallback=DEFAULT_VIDEO_SOURCE)
    enable_masked = config.getboolean("region_masker", "enable_masked_regions", fallback=False)
    masked_json  = config.get("region_masker", "masked_zones_output_json", fallback=DEFAULT_MASKED_ZONES_JSON)
    enable_named = config.getboolean("region_masker", "enable_zone_names", fallback=False)
    named_json   = config.get("region_masker", "named_zones_output_json", fallback=DEFAULT_NAMED_ZONES_JSON)
    critical_flag = config.getboolean("region_masker", "use_critical_thresholds", fallback=False)

    logger.info("=== [region_masker] settings from config.ini ===")
    logger.info(f"video_source={video_source}")
    logger.info(f"enable_masked_regions={enable_masked}, masked_zones_output_json={masked_json}")
    logger.info(f"enable_zone_names={enable_named}, named_zones_output_json={named_json}")
    logger.info(f"use_critical_thresholds={critical_flag}")
    logger.info("===============================================")

    if enable_masked and not os.path.exists(masked_json):
        fallback_masked = os.path.join("data", os.path.basename(masked_json))
        if os.path.exists(fallback_masked):
            logger.info(f"Using fallback path for masked_zones_output_json: {fallback_masked}")
            masked_json = fallback_masked
        else:
            logger.error(f"Requested to enable masked regions, but JSON not found: {masked_json}")

    if enable_named and not os.path.exists(named_json):
        fallback_named = os.path.join("data", os.path.basename(named_json))
        if os.path.exists(fallback_named):
            logger.info(f"Using fallback path for named_zones_output_json: {fallback_named}")
            named_json = fallback_named
        else:
            logger.error(f"Requested to enable named zones, but JSON not found: {named_json}")

    return (
        True,
        video_source,
        enable_masked,
        masked_json,
        enable_named,
        named_json,
        critical_flag
    )

def scale_frame_if_needed(frame):
    global original_width, original_height, display_width, display_height
    original_height, original_width = frame.shape[:2]
    display_width = original_width
    display_height = original_height

    if display_width > MAX_DISPLAY_WIDTH or display_height > MAX_DISPLAY_HEIGHT:
        scale_w = MAX_DISPLAY_WIDTH / float(display_width)
        scale_h = MAX_DISPLAY_HEIGHT / float(display_height)
        scale = min(scale_w, scale_h)
        new_w = int(display_width * scale)
        new_h = int(display_height * scale)
        logger.info(f"Resizing display from {display_width}x{display_height} to {new_w}x{new_h}")
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        display_width, display_height = new_w, new_h

    return frame

def to_original_coords(x_disp, y_disp):
    if display_width == original_width and display_height == original_height:
        return x_disp, y_disp
    scale_x = original_width / float(display_width)
    scale_y = original_height / float(display_height)
    return int(x_disp * scale_x), int(y_disp * scale_y)

def to_display_coords(x_orig, y_orig):
    if display_width == original_width and display_height == original_height:
        return x_orig, y_orig
    scale_x = display_width / float(original_width)
    scale_y = display_height / float(original_height)
    return int(x_orig * scale_x), int(y_orig * scale_y)

def save_zones_to_json(output_json):
    dir_path = Path(output_json).parent
    if dir_path and not dir_path.exists():
        logger.info(f"Output directory '{dir_path}' does not exist; creating it.")
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Unable to create directory '{dir_path}': {e}")
            sys.exit("Cannot create output directory for zones JSON.")

    if zone_mode == "masked":
        data_key = "masked_zones"
    else:
        data_key = "named_zones"

    logger.info(f"Saving {zone_mode.upper()} zones to {output_json} ...")
    with open(output_json, 'w') as f:
        json.dump({data_key: zones}, f, indent=4)
    logger.info("Zones saved successfully.")

def draw_rectangles(event, x_disp, y_disp, flags, param):
    global ix, iy, drawing, temp_rect, use_critical_thresholds, zone_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x_disp, y_disp
        temp_rect = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_rect = (ix, iy, x_disp, y_disp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if temp_rect:
            x1_disp, y1_disp, x2_disp, y2_disp = temp_rect
            x1_disp, x2_disp = sorted([x1_disp, x2_disp])
            y1_disp, y2_disp = sorted([y1_disp, y2_disp])

            x1_orig, y1_orig = to_original_coords(x1_disp, y1_disp)
            x2_orig, y2_orig = to_original_coords(x2_disp, y2_disp)

            default_name = f"Zone_{len(zones) + 1}"
            zone_name = input(f"Enter zone name (press Enter for '{default_name}'): ").strip()
            if not zone_name:
                zone_name = default_name

            new_zone = {
                "name": zone_name,
                "x1": x1_orig, "y1": y1_orig,
                "x2": x2_orig, "y2": y2_orig
            }

            if zone_mode == "masked":
                while True:
                    conf_str = input("Enter min confidence threshold (0.0–1.0, Enter=0.0): ").strip()
                    if not conf_str:
                        zone_conf = 0.0
                        break
                    try:
                        zone_conf = float(conf_str)
                        if 0.0 <= zone_conf <= 1.0:
                            break
                        else:
                            print("Please enter a number between 0.0 and 1.0.")
                    except ValueError:
                        print("Invalid input, please enter a numeric value.")

                new_zone["confidence_threshold"] = zone_conf
                logger.info(
                    f"Added MASKED zone '{zone_name}' with min conf={zone_conf:.2f}, "
                    f"orig=({x1_orig},{y1_orig})-({x2_orig},{y2_orig})"
                )

            elif zone_mode == "named":
                if use_critical_thresholds:
                    crit_str = input("Enter critical threshold (0.0–1.0, Enter=none): ").strip()
                    if crit_str:
                        try:
                            cval = float(crit_str)
                            if 0.0 <= cval <= 1.0:
                                new_zone["critical_threshold"] = cval
                                logger.info(f"  -> CRITICAL threshold set to {cval:.2f}")
                            else:
                                logger.info(f"Value '{crit_str}' is out of 0.0–1.0; ignoring.")
                        except ValueError:
                            logger.info(f"Invalid input '{crit_str}' for critical threshold; ignoring.")

                logger.info(
                    f"Added NAMED zone '{zone_name}', orig=({x1_orig},{y1_orig})-({x2_orig},{y2_orig})"
                )

            zones.append(new_zone)

            # We still offer an immediate chance to save if they want:
            ans = input("Would you like to SAVE the zones now? (y/N) ").strip().lower()
            if ans == 'y':
                save_zones_to_json(zone_output_file)

        temp_rect = None

def main():
    global zone_mode, zone_output_file, zones
    global use_critical_thresholds

    (
        config_found,
        video_source,
        enable_masked,
        masked_json,
        enable_named,
        named_json,
        critical_flag
    ) = load_config()

    use_critical_thresholds = critical_flag

    if not config_found:
        logger.warning("No usable config.ini found. Using default/fallback settings.")
    else:
        logger.info("Config.ini loaded successfully.")

    if not enable_masked:
        logger.warning("Heads-up: 'enable_masked_regions' is FALSE. You can still create masked zones, but main pipeline won't use them unless enabled!")
    if not enable_named:
        logger.warning("Heads-up: 'enable_zone_names' is FALSE. You can still create named zones, but main pipeline won't label them unless enabled!")

    print("Which type of zones do you want to draw?")
    print("   [1] Masked zones (with min confidence thresholds)")
    print("   [2] Named zones (with optional critical thresholds)")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            zone_mode = "masked"
            zone_output_file = masked_json
            break
        elif choice == "2":
            zone_mode = "named"
            zone_output_file = named_json
            break
        else:
            print("Invalid input. Please enter '1' or '2'.")

    if not os.path.exists(zone_output_file):
        logger.info(f"No existing {zone_mode} zones file found at '{zone_output_file}'. We'll create it when you save.")
    else:
        logger.info(f"Found existing {zone_mode} zones file: {zone_output_file}, attempting to load.")
        try:
            with open(zone_output_file, 'r') as f:
                data = json.load(f)
            if zone_mode == "masked":
                loaded_z = data.get("masked_zones", [])
            else:
                loaded_z = data.get("named_zones", [])

            zones.extend(loaded_z)
            logger.info(f"Loaded {len(loaded_z)} {zone_mode} zones from {zone_output_file}.")
        except Exception as e:
            logger.error(f"Failed to parse JSON from {zone_output_file}: {e}")
            logger.info("Continuing with an empty list.")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {video_source}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.error("Could not read a frame from the stream.")
        return

    frame_display = scale_frame_if_needed(frame)

    window_title = f"Draw {zone_mode.upper()} Zones"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, draw_rectangles)

    try:
        while True:
            display_frame = frame_display.copy()

            # Choose color for each mode
            if zone_mode == "masked":
                color = (0, 0, 255)  # red
                prefix = "[MASKED]"
            else:
                color = (0, 255, 0)  # green
                prefix = "[NAMED]"

            # Draw existing zones
            for z in zones:
                x1o, y1o = z.get("x1", 0), z.get("y1", 0)
                x2o, y2o = z.get("x2", 0), z.get("y2", 0)
                x1d, y1d = to_display_coords(x1o, y1o)
                x2d, y2d = to_display_coords(x2o, y2o)
                cv2.rectangle(display_frame, (x1d, y1d), (x2d, y2d), color, 2)

                name = z.get("name", "?")

                if zone_mode == "masked":
                    cthr = z.get("confidence_threshold", 0.0)
                    label = f"{prefix} {name} (min {cthr:.2f})"
                else:
                    cval = z.get("critical_threshold", None)
                    if cval is not None:
                        label = f"{prefix} {name} (crit≥{cval:.2f})"
                    else:
                        label = f"{prefix} {name}"

                label_y = y1d - 5 if (y1d - 10) > 0 else y1d + 15
                cv2.putText(display_frame, label, (x1d, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if temp_rect:
                x1d, y1d, x2d, y2d = temp_rect
                cv2.rectangle(display_frame, (x1d, y1d), (x2d, y2d), (255,255,0), 2)

            cv2.putText(
                display_frame,
                f"Mode: {zone_mode.upper()} - L-click & drag. Press 's' to save, 'q' to quit.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(window_title, display_frame)

            # If user closed the window via the X button, handle it:
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Window closed by user. Exiting main loop.")
                break

            key = cv2.waitKey(20) & 0xFF
            if key == ord('s'):
                save_zones_to_json(zone_output_file)
            elif key == ord('q'):
                logger.info("User pressed 'q'. Exiting main loop.")
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt (Ctrl-C) caught, exiting main loop.")
        pass
    finally:
        # If the loop was broken by 'q', window close, or ctrl-c, we do a final prompt
        cv2.destroyAllWindows()
        ask = input("Save changes before exiting? (y/N) ").strip().lower()
        if ask == 'y':
            save_zones_to_json(zone_output_file)
        else:
            logger.info("No changes saved. Goodbye!")

if __name__ == "__main__":
    main()
