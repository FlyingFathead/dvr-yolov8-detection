#!/usr/bin/env python3
# File: ./utils/region_masker.py

import cv2
import json
import os
import sys
import configparser
import logging
from pathlib import Path

# ---------------------------------------------------------------------
# Default fallback settings
# ---------------------------------------------------------------------
DEFAULT_VIDEO_SOURCE = "rtmp://127.0.0.1:1935/live/stream"
DEFAULT_MASKED_ZONES_JSON = "ignore_zones.json"
DEFAULT_NAMED_ZONES_JSON = "named_zones.json"

# If the captured frame is bigger than this, we'll scale it down for display only
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

# We'll store whichever set of zones (masked or named) in a single list:
zones = []

# Keep track of the original vs. displayed size so we can transform coords
original_width = None
original_height = None
display_width = None
display_height = None

drawing = False
ix, iy = -1, -1
temp_rect = None

zone_mode = None         # "masked" or "named"
zone_output_file = None  # Which JSON file we write to

# ---------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------
logger = logging.getLogger("region_masker")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

# ---------------------------------------------------------------------
# Load config.ini ([region_masker] section)
# ---------------------------------------------------------------------
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
            DEFAULT_NAMED_ZONES_JSON
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
            DEFAULT_NAMED_ZONES_JSON
        )

    video_source = config.get("region_masker", "video_source", fallback=DEFAULT_VIDEO_SOURCE)
    enable_masked = config.getboolean("region_masker", "enable_masked_regions", fallback=False)
    masked_json  = config.get("region_masker", "masked_zones_output_json", fallback=DEFAULT_MASKED_ZONES_JSON)
    enable_named = config.getboolean("region_masker", "enable_zone_names", fallback=False)
    named_json   = config.get("region_masker", "named_zones_output_json", fallback=DEFAULT_NAMED_ZONES_JSON)

    logger.info("=== [region_masker] settings from config.ini ===")
    logger.info(f"video_source={video_source}")
    logger.info(f"enable_masked_regions={enable_masked}, masked_zones_output_json={masked_json}")
    logger.info(f"enable_zone_names={enable_named}, named_zones_output_json={named_json}")
    logger.info("===============================================")

    # If user specified "ignore_zones.json" or something that isn't found, check for fallback in ./data/
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
        True,  # means config found
        video_source,
        enable_masked,
        masked_json,
        enable_named,
        named_json
    )


# ---------------------------------------------------------------------
# Possibly downscale the frame for display
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Convert display coords -> original coords
# ---------------------------------------------------------------------
def to_original_coords(x_disp, y_disp):
    if display_width == original_width and display_height == original_height:
        return x_disp, y_disp
    scale_x = original_width / float(display_width)
    scale_y = original_height / float(display_height)
    return int(x_disp * scale_x), int(y_disp * scale_y)

# ---------------------------------------------------------------------
# Convert original coords -> display coords
# ---------------------------------------------------------------------
def to_display_coords(x_orig, y_orig):
    if display_width == original_width and display_height == original_height:
        return x_orig, y_orig
    scale_x = display_width / float(original_width)
    scale_y = display_height / float(original_height)
    return int(x_orig * scale_x), int(y_orig * scale_y)

# ---------------------------------------------------------------------
# Save to JSON
# ---------------------------------------------------------------------
def save_zones_to_json(output_json):
    """Writes current `zones` to a JSON file under "masked_zones" or "named_zones", 
       depending on zone_mode."""
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

# ---------------------------------------------------------------------
# Mouse callback: handle rectangle drawing & user input
# ---------------------------------------------------------------------
def draw_rectangles(event, x_disp, y_disp, flags, param):
    global ix, iy, drawing, temp_rect

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
            # Normalize coords
            x1_disp, x2_disp = sorted([x1_disp, x2_disp])
            y1_disp, y2_disp = sorted([y1_disp, y2_disp])

            # Convert from display coords to original coords
            x1_orig, y1_orig = to_original_coords(x1_disp, y1_disp)
            x2_orig, y2_orig = to_original_coords(x2_disp, y2_disp)

            default_name = f"Zone_{len(zones) + 1}"
            zone_name = input(f"Enter zone name (press Enter for '{default_name}'): ").strip()
            if not zone_name:
                zone_name = default_name

            # Prompt for the min confidence threshold
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

            # Build the new zone dictionary
            new_zone = {
                "name": zone_name,
                "confidence_threshold": zone_conf,
                "x1": x1_orig, "y1": y1_orig,
                "x2": x2_orig, "y2": y2_orig
            }
            zones.append(new_zone)
            logger.info(
                f"Added {zone_mode.upper()} zone '{zone_name}' with conf={zone_conf:.2f}, "
                f"orig=({x1_orig},{y1_orig})-({x2_orig},{y2_orig}), "
                f"display=({x1_disp},{y1_disp})-({x2_disp},{y2_disp})"
            )

            # Prompt to save right away
            ans = input("Would you like to SAVE the zones now? (y/N) ").strip().lower()
            if ans == 'y':
                save_zones_to_json(zone_output_file)

        temp_rect = None


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    global zone_mode, zone_output_file, zones

    # 1) Load config, check if it was found
    (
        config_found,
        video_source,
        enable_masked,
        masked_json,
        enable_named,
        named_json
    ) = load_config()

    if not config_found:
        logger.warning("No usable config.ini found. Using default/fallback settings.")
    else:
        logger.info("Config.ini loaded successfully.")

    # If config says masked is disabled, just warn:
    if not enable_masked:
        logger.warning("Heads-up: 'enable_masked_regions' is FALSE in config.ini. "
                       "You can still create masked zones here, but the main pipeline won't use them "
                       "unless you enable it in config.ini!")
    if not enable_named:
        logger.warning("Heads-up: 'enable_zone_names' is FALSE in config.ini. "
                       "You can still create named zones here, but the main pipeline won't label them "
                       "unless you enable it in config.ini!")

    # We no longer forcibly exit if both are false—just warn the user.
    # 2) Let the user pick which mode they want to draw:
    print("Which type of zones do you want to draw?")
    print("   [1] Masked zones (with min confidence thresholds)")
    print("   [2] Named zones (for labeling detections)")
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

    # 3) If the chosen JSON doesn't exist yet, we won't forcibly fail—just note it:
    if not os.path.exists(zone_output_file):
        logger.info(f"No existing {zone_mode} zones file found at '{zone_output_file}'. "
                    "We'll create it when you save.")

    else:
        # Attempt to load existing zones
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
            logger.info(f"Continuing with an empty {zone_mode} zones list.")

    # 4) Open the video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {video_source}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.error("Could not read a frame from the stream.")
        return

    # 5) Possibly scale it
    frame_display = scale_frame_if_needed(frame)

    # 6) Setup window + callback
    window_title = f"Draw {zone_mode.upper()} Zones"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, draw_rectangles)

    # 7) Main loop
    while True:
        display_frame = frame_display.copy()

        # Decide color & label prefix
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
            cthr = z.get("confidence_threshold", 0.0)
            label = f"{prefix} {name} ({cthr:.2f})"
            label_y = y1d - 5 if (y1d - 10) > 0 else y1d + 15
            cv2.putText(display_frame, label, (x1d, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # If user is currently drawing a rect
        if temp_rect:
            x1d, y1d, x2d, y2d = temp_rect
            cv2.rectangle(display_frame, (x1d, y1d), (x2d, y2d), (255,255,0), 2)

        # Info text
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
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            save_zones_to_json(zone_output_file)
        elif key == ord('q'):
            ans = input("You pressed 'q'. Save changes before exiting? (y/N) ").strip().lower()
            if ans == 'y':
                save_zones_to_json(zone_output_file)
            break

    cv2.destroyAllWindows()

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# # File: ./utils/region_masker.py

# import cv2
# import json
# import os
# import sys
# import configparser
# import logging

# from pathlib import Path

# # --------------------------
# # Default fallback settings:
# # --------------------------
# DEFAULT_VIDEO_SOURCE = "rtmp://127.0.0.1:1935/live/stream"
# DEFAULT_IGNORED_ZONES_JSON = "ignore_zones.json"
# DEFAULT_NAMED_ZONES_JSON = "named_zones.json"

# # If the captured frame is bigger than this, we'll scale it down for display only
# MAX_DISPLAY_WIDTH = 1280
# MAX_DISPLAY_HEIGHT = 720

# # We'll store the current mode's zones in a single list: "zones"
# zones = []

# # Keep track of the original vs. displayed size so we can transform coordinates
# original_width = None
# original_height = None
# display_width = None
# display_height = None

# drawing = False
# ix, iy = -1, -1  # mouse-down coordinates (in display coords)
# temp_rect = None  # rectangle while dragging (in display coords)

# zone_mode = None         # "ignore" or "named"
# zone_output_file = None  # Which JSON file is currently being edited?

# # ------------------------------------------------------------------------------
# # Logger setup
# # ------------------------------------------------------------------------------
# logger = logging.getLogger("region_masker")
# logger.setLevel(logging.INFO)
# ch = logging.StreamHandler(sys.stdout)
# ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
# logger.addHandler(ch)

# # ------------------------------------------------------------------------------
# # Load config.ini ([region_masker] section)
# # ------------------------------------------------------------------------------
# def load_config():
#     config = configparser.ConfigParser()
#     found = config.read("config.ini")

#     if not found:
#         logger.warning("No config.ini found or it's empty. Using default settings.")
#         return (
#             DEFAULT_VIDEO_SOURCE,
#             True,  # pretend enable_masked
#             DEFAULT_IGNORED_ZONES_JSON,
#             True,  # pretend enable_named
#             DEFAULT_NAMED_ZONES_JSON
#         )

#     if not config.has_section("region_masker"):
#         logger.warning("[region_masker] section not found. Using default settings.")
#         return (
#             DEFAULT_VIDEO_SOURCE,
#             True,
#             DEFAULT_IGNORED_ZONES_JSON,
#             True,
#             DEFAULT_NAMED_ZONES_JSON
#         )

#     video_source = config.get("region_masker", "video_source", fallback=DEFAULT_VIDEO_SOURCE)
#     enable_masked = config.getboolean("region_masker", "enable_masked_regions", fallback=False)
#     ignored_json = config.get("region_masker", "ignored_zones_output_json", fallback=DEFAULT_IGNORED_ZONES_JSON)
#     enable_named = config.getboolean("region_masker", "enable_zone_names", fallback=False)
#     named_json = config.get("region_masker", "named_zones_output_json", fallback=DEFAULT_NAMED_ZONES_JSON)

#     logger.info(f"video_source={video_source}")
#     logger.info(f"enable_masked_regions={enable_masked}, ignored_zones_output_json={ignored_json}")
#     logger.info(f"enable_zone_names={enable_named}, named_zones_output_json={named_json}")

#     return (video_source, enable_masked, ignored_json, enable_named, named_json)

# # ------------------------------------------------------------------------------
# # scale_frame_if_needed
# # ------------------------------------------------------------------------------
# def scale_frame_if_needed(frame):
#     global original_width, original_height, display_width, display_height
#     original_height, original_width = frame.shape[:2]
#     display_width = original_width
#     display_height = original_height

#     if display_width > MAX_DISPLAY_WIDTH or display_height > MAX_DISPLAY_HEIGHT:
#         scale_w = MAX_DISPLAY_WIDTH / float(display_width)
#         scale_h = MAX_DISPLAY_HEIGHT / float(display_height)
#         scale = min(scale_w, scale_h)
#         new_w = int(display_width * scale)
#         new_h = int(display_height * scale)
#         logger.info(f"Resizing display from {display_width}x{display_height} to {new_w}x{new_h}")
#         frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
#         display_width, display_height = new_w, new_h

#     return frame

# # ------------------------------------------------------------------------------
# # to_original_coords / to_display_coords
# # ------------------------------------------------------------------------------
# def to_original_coords(x_disp, y_disp):
#     if display_width == original_width and display_height == original_height:
#         return x_disp, y_disp
#     scale_x = original_width / float(display_width)
#     scale_y = original_height / float(display_height)
#     return int(x_disp * scale_x), int(y_disp * scale_y)

# def to_display_coords(x_orig, y_orig):
#     if display_width == original_width and display_height == original_height:
#         return x_orig, y_orig
#     scale_x = display_width / float(original_width)
#     scale_y = display_height / float(original_height)
#     return int(x_orig * scale_x), int(y_orig * scale_y)

# # ------------------------------------------------------------------------------
# # save_zones_to_json
# # ------------------------------------------------------------------------------
# def save_zones_to_json(output_json):
#     dir_path = Path(output_json).parent
#     if dir_path and not dir_path.exists():
#         logger.info(f"Output directory '{dir_path}' does not exist; creating it.")
#         try:
#             dir_path.mkdir(parents=True, exist_ok=True)
#         except Exception as e:
#             logger.error(f"Unable to create directory '{dir_path}': {e}")
#             sys.exit("Cannot create output directory.")

#     if zone_mode == "ignore":
#         # We'll store them under "ignore_zones"
#         data_key = "ignore_zones"
#     else:
#         # We'll store them under "named_zones"
#         data_key = "named_zones"

#     logger.info(f"Saving {zone_mode} zones to {output_json} ...")
#     with open(output_json, 'w') as f:
#         json.dump({data_key: zones}, f, indent=4)
#     logger.info("Zones saved successfully.")

# # ------------------------------------------------------------------------------
# # Mouse callback
# # ------------------------------------------------------------------------------
# def draw_rectangles(event, x_disp, y_disp, flags, param):
#     global ix, iy, drawing, temp_rect

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x_disp, y_disp
#         temp_rect = None

#     elif event == cv2.EVENT_MOUSEMOVE and drawing:
#         temp_rect = (ix, iy, x_disp, y_disp)

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         if temp_rect:
#             x1_disp, y1_disp, x2_disp, y2_disp = temp_rect
#             # Normalize
#             x1_disp, x2_disp = sorted([x1_disp, x2_disp])
#             y1_disp, y2_disp = sorted([y1_disp, y2_disp])

#             # Convert to original coords
#             x1_orig, y1_orig = to_original_coords(x1_disp, y1_disp)
#             x2_orig, y2_orig = to_original_coords(x2_disp, y2_disp)

#             default_name = f"Zone_{len(zones) + 1}"
#             zone_name = input(f"Enter zone name (press Enter for '{default_name}'): ").strip()
#             if not zone_name:
#                 zone_name = default_name

#             # Prompt for confidence threshold
#             while True:
#                 confidence_str = input("Enter min confidence threshold (0.0–1.0, Enter=0.0): ").strip()
#                 if not confidence_str:
#                     zone_conf = 0.0
#                     break
#                 try:
#                     zone_conf = float(confidence_str)
#                     if 0.0 <= zone_conf <= 1.0:
#                         break
#                     else:
#                         print("Please enter a number between 0.0 and 1.0.")
#                 except ValueError:
#                     print("Invalid input, please enter a numeric value.")

#             # Add zone
#             new_zone = {
#                 "name": zone_name,
#                 "confidence_threshold": zone_conf,
#                 "x1": x1_orig, "y1": y1_orig,
#                 "x2": x2_orig, "y2": y2_orig
#             }
#             zones.append(new_zone)
#             logger.info(f"Added {zone_mode.upper()} zone '{zone_name}' w/conf={zone_conf}, "
#                         f"orig=({x1_orig},{y1_orig})-({x2_orig},{y2_orig}), "
#                         f"display=({x1_disp},{y1_disp})-({x2_disp},{y2_disp})")

#             # Prompt to save immediately
#             ans = input("Would you like to SAVE the zones now? (y/N) ").strip().lower()
#             if ans == 'y':
#                 save_zones_to_json(zone_output_file)

#         temp_rect = None

# # ------------------------------------------------------------------------------
# # main
# # ------------------------------------------------------------------------------
# def main():
#     global zone_mode, zone_output_file

#     (video_source,
#      enable_masked, ignored_json,
#      enable_named, named_json) = load_config()

#     # 1) Decide zone_mode
#     # if both are enabled, ask user
#     # if only one is enabled, pick it automatically
#     # if none are enabled, exit
#     if not enable_masked and not enable_named:
#         logger.error("Neither masked nor named zones are enabled. Exiting.")
#         return
#     elif enable_masked and enable_named:
#         print("Which type of zones do you want to draw?")
#         print("   [1] Ignored (masked) zones")
#         print("   [2] Named zones")
#         while True:
#             choice = input("Enter 1 or 2: ").strip()
#             if choice == "1":
#                 zone_mode = "ignore"
#                 zone_output_file = ignored_json
#                 break
#             elif choice == "2":
#                 zone_mode = "named"
#                 zone_output_file = named_json
#                 break
#             else:
#                 print("Invalid input. Please enter '1' or '2'.")
#     elif enable_masked:
#         zone_mode = "ignore"
#         zone_output_file = ignored_json
#         logger.info("Only 'ignored' zones are enabled, proceeding with ignore-mode.")
#     else:
#         zone_mode = "named"
#         zone_output_file = named_json
#         logger.info("Only 'named' zones are enabled, proceeding with named-mode.")

#     # 2) Attempt to load existing zones from that file (if it exists)
#     if os.path.exists(zone_output_file):
#         logger.info(f"Found existing {zone_mode} zones file: {zone_output_file}, attempting to load.")
#         try:
#             with open(zone_output_file, 'r') as f:
#                 data = json.load(f)
#             if zone_mode == "ignore":
#                 loaded = data.get("ignore_zones", [])
#             else:
#                 loaded = data.get("named_zones", [])
#             zones.extend(loaded)
#             logger.info(f"Loaded {len(loaded)} {zone_mode} zones from {zone_output_file}.")
#         except Exception as e:
#             logger.error(f"Failed to parse JSON from {zone_output_file}: {e}")
#             logger.info(f"Continuing with an empty {zone_mode} zones list.")
#     else:
#         logger.info(f"No existing {zone_mode} zones file at '{zone_output_file}'. Starting empty.")

#     # 3) Try opening the video source
#     cap = cv2.VideoCapture(video_source)
#     if not cap.isOpened():
#         logger.error(f"Could not open video source: {video_source}")
#         return

#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         logger.error("Could not read a frame from the stream.")
#         return

#     # 4) Possibly scale it for display
#     frame_display = scale_frame_if_needed(frame)

#     # 5) Setup window + callback
#     window_title = f"Draw {zone_mode.upper()} Zones"
#     cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
#     cv2.setMouseCallback(window_title, draw_rectangles)

#     # 6) Main loop
#     while True:
#         display_frame = frame_display.copy()

#         # Draw existing zones (different color for each mode if you want)
#         if zone_mode == "ignore":
#             color = (0, 0, 255)   # red
#             prefix = "[IGNORE]"
#         else:
#             color = (0, 255, 0)   # green
#             prefix = "[NAMED]"

#         for z in zones:
#             x1o, y1o = z.get("x1", 0), z.get("y1", 0)
#             x2o, y2o = z.get("x2", 0), z.get("y2", 0)
#             x1d, y1d = to_display_coords(x1o, y1o)
#             x2d, y2d = to_display_coords(x2o, y2o)
#             cv2.rectangle(display_frame, (x1d, y1d), (x2d, y2d), color, 2)
#             name = z.get("name", "?")
#             cthr = z.get("confidence_threshold", 0.0)
#             label = f"{prefix} {name} ({cthr})"
#             label_y = y1d - 5 if (y1d - 10) > 0 else y1d + 15
#             cv2.putText(display_frame, label, (x1d, label_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

#         # If user is currently drawing, show the "temp_rect" in cyan
#         if temp_rect:
#             x1d, y1d, x2d, y2d = temp_rect
#             cv2.rectangle(display_frame, (x1d, y1d), (x2d, y2d), (255,255,0), 2)

#         # Info text
#         cv2.putText(display_frame,
#                     f"Mode: {zone_mode.upper()} - Left-click & drag. Press 's' to save, 'q' to quit.",
#                     (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (255, 255, 255),
#                     2)

#         cv2.imshow(window_title, display_frame)
#         key = cv2.waitKey(20) & 0xFF
#         if key == ord('s'):
#             save_zones_to_json(zone_output_file)
#         elif key == ord('q'):
#             ans = input("You pressed 'q'. Save changes before exiting? (y/N) ").strip().lower()
#             if ans == 'y':
#                 save_zones_to_json(zone_output_file)
#             break

#     cv2.destroyAllWindows()

# # ------------------------------------------------------------------------------
# if __name__ == "__main__":
#     main()

# # #  //
# # # // old; tbd
# # #!/usr/bin/env python3
# # # File: ./utils/region_masker.py

# # import cv2
# # import json
# # import os
# # import sys
# # import configparser
# # import logging

# # from pathlib import Path

# # # --------------------------
# # # Default fallback settings:
# # # --------------------------
# # DEFAULT_VIDEO_SOURCE = "rtmp://127.0.0.1:1935/live/stream"
# # DEFAULT_OUTPUT_JSON = "ignore_zones.json"

# # # If the captured frame is bigger than this, we'll scale it down for display only
# # MAX_DISPLAY_WIDTH = 1280
# # MAX_DISPLAY_HEIGHT = 720

# # # Each element: {"name": str, "confidence_threshold": float, "x1": int, "y1": int, "x2": int, "y2": int}
# # zones = []

# # # We’ll keep track of the original vs. displayed size so we can transform coordinates back/forth.
# # original_width = None
# # original_height = None
# # display_width = None
# # display_height = None

# # drawing = False
# # ix, iy = -1, -1  # mouse-down coordinates (in display coords)
# # temp_rect = None  # rectangle while dragging (in display coords)

# # # ------------------------------------------------------------------------------
# # # Logger setup
# # # ------------------------------------------------------------------------------
# # logger = logging.getLogger("region_masker")
# # logger.setLevel(logging.INFO)
# # ch = logging.StreamHandler(sys.stdout)
# # ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
# # logger.addHandler(ch)

# # # ------------------------------------------------------------------------------
# # # Load config.ini ([region_masker] section)
# # # ------------------------------------------------------------------------------
# # def load_config():
# #     """Load 'video_source' and 'output_json' from config.ini if present,
# #        otherwise fallback to defaults and log warnings."""
# #     config = configparser.ConfigParser()
# #     read_files = config.read("config.ini")

# #     if not read_files:
# #         logger.warning("No config.ini found or it's empty. Using default settings.")
# #         return DEFAULT_VIDEO_SOURCE, DEFAULT_OUTPUT_JSON

# #     if not config.has_section("region_masker"):
# #         logger.warning("[region_masker] section not found in config.ini. Using defaults.")
# #         return DEFAULT_VIDEO_SOURCE, DEFAULT_OUTPUT_JSON

# #     video_source = config.get("region_masker", "video_source", fallback=DEFAULT_VIDEO_SOURCE)
# #     output_json = config.get("region_masker", "output_json", fallback=DEFAULT_OUTPUT_JSON)

# #     logger.info(f"Using video_source: {video_source}")
# #     logger.info(f"Using output_json: {output_json}")
# #     return video_source, output_json

# # # ------------------------------------------------------------------------------
# # # Scale the frame if it's larger than our max display size
# # # ------------------------------------------------------------------------------
# # def scale_frame_if_needed(frame):
# #     """Returns a possibly downscaled copy for display, sets global width/height."""
# #     global original_width, original_height, display_width, display_height

# #     original_height, original_width = frame.shape[:2]
# #     display_width = original_width
# #     display_height = original_height

# #     if display_width > MAX_DISPLAY_WIDTH or display_height > MAX_DISPLAY_HEIGHT:
# #         scale_w = MAX_DISPLAY_WIDTH / float(display_width)
# #         scale_h = MAX_DISPLAY_HEIGHT / float(display_height)
# #         scale = min(scale_w, scale_h)
# #         new_w = int(display_width * scale)
# #         new_h = int(display_height * scale)

# #         logger.info(f"Resizing display from {display_width}x{display_height} to {new_w}x{new_h}")
# #         frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

# #         display_width, display_height = new_w, new_h

# #     return frame

# # # ------------------------------------------------------------------------------
# # # Convert display coords -> original coords
# # # ------------------------------------------------------------------------------
# # def to_original_coords(x_disp, y_disp):
# #     """Given (x_disp,y_disp) in the display, return the corresponding original coords."""
# #     if display_width == original_width and display_height == original_height:
# #         return x_disp, y_disp
# #     scale_x = original_width / float(display_width)
# #     scale_y = original_height / float(display_height)
# #     return int(x_disp * scale_x), int(y_disp * scale_y)

# # # ------------------------------------------------------------------------------
# # # Convert original coords -> display coords
# # # ------------------------------------------------------------------------------
# # def to_display_coords(x_orig, y_orig):
# #     """Given (x_orig,y_orig) in the original, return the corresponding display coords."""
# #     if display_width == original_width and display_height == original_height:
# #         return x_orig, y_orig
# #     scale_x = display_width / float(original_width)
# #     scale_y = display_height / float(original_height)
# #     return int(x_orig * scale_x), int(y_orig * scale_y)

# # # ------------------------------------------------------------------------------
# # # Save zones to output JSON
# # # ------------------------------------------------------------------------------
# # def save_zones_to_json(output_json):
# #     """Writes current `zones` to a JSON file, creating/overwriting it."""
# #     dir_path = Path(output_json).parent
# #     if dir_path and not dir_path.exists():
# #         logger.info(f"Output directory '{dir_path}' does not exist; attempting to create it.")
# #         try:
# #             dir_path.mkdir(parents=True, exist_ok=True)
# #         except Exception as e:
# #             logger.error(f"Unable to create directory '{dir_path}': {e}")
# #             sys.exit("Error: cannot create output directory. Please fix config.ini or your path.")

# #     logger.info(f"Saving zones to {output_json} ...")
# #     with open(output_json, 'w') as f:
# #         json.dump({"ignore_zones": zones}, f, indent=4)
# #     logger.info("Zones saved successfully.")

# # # ------------------------------------------------------------------------------
# # # Mouse callback
# # # ------------------------------------------------------------------------------
# # def draw_rectangles(event, x_disp, y_disp, flags, param):
# #     global ix, iy, drawing, temp_rect

# #     if event == cv2.EVENT_LBUTTONDOWN:
# #         drawing = True
# #         ix, iy = x_disp, y_disp
# #         temp_rect = None

# #     elif event == cv2.EVENT_MOUSEMOVE and drawing:
# #         temp_rect = (ix, iy, x_disp, y_disp)

# #     elif event == cv2.EVENT_LBUTTONUP:
# #         drawing = False
# #         if temp_rect:
# #             x1_disp, y1_disp, x2_disp, y2_disp = temp_rect
# #             # Normalize
# #             x1_disp, x2_disp = sorted([x1_disp, x2_disp])
# #             y1_disp, y2_disp = sorted([y1_disp, y2_disp])

# #             # Convert to original coords
# #             x1_orig, y1_orig = to_original_coords(x1_disp, y1_disp)
# #             x2_orig, y2_orig = to_original_coords(x2_disp, y2_disp)

# #             # Prompt for zone name
# #             default_name = f"Zone_{len(zones) + 1}"
# #             zone_name = input(f"Enter zone name (press Enter for '{default_name}'): ").strip()
# #             if not zone_name:
# #                 zone_name = default_name

# #             # Prompt for confidence
# #             while True:
# #                 confidence_str = input("Enter min confidence threshold (0.0–1.0, Enter=0.0): ").strip()
# #                 if not confidence_str:
# #                     zone_conf = 0.0
# #                     break
# #                 try:
# #                     zone_conf = float(confidence_str)
# #                     if 0.0 <= zone_conf <= 1.0:
# #                         break
# #                     else:
# #                         print("Please enter a number between 0.0 and 1.0.")
# #                 except ValueError:
# #                     print("Invalid input, please enter a numeric value.")

# #             # Add zone
# #             zones.append({
# #                 "name": zone_name,
# #                 "confidence_threshold": zone_conf,
# #                 "x1": x1_orig, "y1": y1_orig,
# #                 "x2": x2_orig, "y2": y2_orig
# #             })
# #             logger.info(f"Added zone '{zone_name}' w/conf={zone_conf}, "
# #                         f"orig=({x1_orig},{y1_orig})-({x2_orig},{y2_orig}), "
# #                         f"display=({x1_disp},{y1_disp})-({x2_disp},{y2_disp})")

# #             # Prompt to save immediately
# #             ans = input("Would you like to SAVE the zones now? (y/N) ").strip().lower()
# #             if ans == 'y':
# #                 save_zones_to_json(param["output_json"])

# #         temp_rect = None

# # # ------------------------------------------------------------------------------
# # # Main
# # # ------------------------------------------------------------------------------
# # def main():
# #     video_source, output_json = load_config()

# #     # 1) Ensure directory for output_json is creatable
# #     dir_path = Path(output_json).parent
# #     if dir_path and not dir_path.exists():
# #         logger.info(f"Output directory '{dir_path}' does not exist; attempting to create it.")
# #         try:
# #             dir_path.mkdir(parents=True, exist_ok=True)
# #         except Exception as e:
# #             logger.error(f"Unable to create directory '{dir_path}': {e}")
# #             sys.exit("Error: cannot create output directory. Please fix config.ini or your path.")

# #     # 2) If output file exists, load existing zones
# #     if os.path.exists(output_json):
# #         logger.info(f"Found existing JSON file: {output_json}, attempting to load.")
# #         try:
# #             with open(output_json, 'r') as f:
# #                 data = json.load(f)
# #             loaded_zones = data.get("ignore_zones", [])
# #             zones.extend(loaded_zones)
# #             logger.info(f"Loaded {len(loaded_zones)} zones from {output_json}.")
# #             for i, z in enumerate(loaded_zones, 1):
# #                 name = z.get("name", f"Zone_{i}")
# #                 cthr = z.get("confidence_threshold", 0.0)
# #                 x1 = z.get("x1", 0)
# #                 y1 = z.get("y1", 0)
# #                 x2 = z.get("x2", 0)
# #                 y2 = z.get("y2", 0)
# #                 logger.info(f"  -> Zone {i}: '{name}', threshold={cthr}, coords=({x1},{y1})-({x2},{y2})")
# #         except Exception as e:
# #             logger.error(f"Failed to parse JSON from {output_json}: {e}")
# #             logger.info("Continuing with an empty zones list.")
# #     else:
# #         logger.info(f"No existing JSON file found at '{output_json}'. Starting with empty list.")

# #     # 3) Open the stream
# #     cap = cv2.VideoCapture(video_source)
# #     if not cap.isOpened():
# #         logger.error(f"Could not open video source: {video_source}")
# #         return

# #     ret, frame = cap.read()
# #     cap.release()
# #     if not ret:
# #         logger.error("Could not read a frame from the stream.")
# #         return

# #     # 4) Possibly scale it for display
# #     frame_display = scale_frame_if_needed(frame)

# #     # 5) Setup window + callback
# #     cv2.namedWindow("Draw Ignore Zones", cv2.WINDOW_NORMAL)
# #     cv2.setMouseCallback("Draw Ignore Zones", draw_rectangles, param={"output_json": output_json})

# #     # 6) Main loop
# #     while True:
# #         display_frame = frame_display.copy()

# #         # Draw existing zones in red
# #         for z in zones:
# #             x1o, y1o = z.get("x1", 0), z.get("y1", 0)
# #             x2o, y2o = z.get("x2", 0), z.get("y2", 0)
# #             x1d, y1d = to_display_coords(x1o, y1o)
# #             x2d, y2d = to_display_coords(x2o, y2o)
# #             cv2.rectangle(display_frame, (x1d, y1d), (x2d, y2d), (0, 0, 255), 2)
# #             name = z.get("name", "?")
# #             cthr = z.get("confidence_threshold", 0.0)
# #             label = f"{name} ({cthr})"
# #             label_y = y1d - 5 if (y1d - 10) > 0 else y1d + 15
# #             cv2.putText(display_frame, label, (x1d, label_y),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# #         # If user is currently drawing, show the "temp_rect"
# #         if temp_rect:
# #             x1d, y1d, x2d, y2d = temp_rect
# #             cv2.rectangle(display_frame, (x1d, y1d), (x2d, y2d), (0,255,0), 2)

# #         # Info text
# #         cv2.putText(display_frame,
# #                     "Left-click & drag to draw. Press 's' to save, 'q' to quit.",
# #                     (10, 30),
# #                     cv2.FONT_HERSHEY_SIMPLEX,
# #                     0.8,
# #                     (255, 255, 255),
# #                     2)

# #         cv2.imshow("Draw Ignore Zones", display_frame)
# #         key = cv2.waitKey(20) & 0xFF
# #         if key == ord('s'):
# #             # Save
# #             save_zones_to_json(output_json)
# #         elif key == ord('q'):
# #             # Prompt user to save or not
# #             ans = input("You pressed 'q'. Do you want to SAVE changes before exiting? (y/N) ").strip().lower()
# #             if ans == 'y':
# #                 save_zones_to_json(output_json)
# #             break

# #     cv2.destroyAllWindows()

# # # ------------------------------------------------------------------------------
# # # Entry Point
# # # ------------------------------------------------------------------------------
# # if __name__ == "__main__":
# #     main()
