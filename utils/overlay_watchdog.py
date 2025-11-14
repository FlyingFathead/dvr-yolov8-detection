#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay OCR watchdog:

- Crops the "watchdog" ROI defined by region_masker (mode 3).
- OCRs the timestamp overlay with Tesseract.
- Parses it using a configurable *human* layout like "DD/MM/YYYY  HH:MM:SS".
- Detects:
    * Frozen overlay time (no change for stuck_threshold_sec).
    * Deadloop (time jumping backwards repeatedly).
    * Optional: overlay lagging behind system time for too long.
- Runs configured shell commands on stuck/loop/error.

Requires:
    pip install opencv-python pytesseract
    tesseract-ocr installed system-wide.
"""

import cv2
import pytesseract
import configparser
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------
# Global switches
# ---------------------------------------------------------------------
# Set this to False once you're happy with behavior
DRY_RUN_MODE = True

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("overlay_watchdog")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
ch.setFormatter(formatter)
logger.addHandler(ch)

# ---------------------------------------------------------------------
# Helpers: timestamp layout -> strptime format
# ---------------------------------------------------------------------

def layout_to_strftime(layout: str) -> str:
    """
    Convert a human layout like "DD/MM/YYYY  HH:MM:SS" into a Python
    strptime format like "%d/%m/%Y  %H:%M:%S".

    Rules:
    - DD -> %d
    - First MM  -> %m (month)
    - Remaining MM -> %M (minutes)
    - YYYY -> %Y
    - YY -> %y
    - HH -> %H
    - SS/ss -> %S
    """
    s = layout

    # Date part
    s = s.replace("YYYY", "%Y")
    s = s.replace("YY", "%y")
    s = s.replace("DD", "%d")

    # First MM we see will be "month"
    if "MM" in s:
        s = s.replace("MM", "%m", 1)

    # Any remaining MM we treat as "minutes"
    s = s.replace("MM", "%M")

    # Hours / seconds
    s = s.replace("HH", "%H")
    s = s.replace("hh", "%I")  # 12h, just in case
    s = s.replace("SS", "%S")
    s = s.replace("ss", "%S")

    return s

# ---------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------
def load_overlay_config(path="config.ini"):
    # Disable interpolation so '%' in format strings never cause trouble
    cfg = configparser.ConfigParser(interpolation=None)
    found = cfg.read(path)
    if not found:
        logger.error("No config.ini found.")
        sys.exit(1)

    if not cfg.has_section("overlay_watchdog"):
        logger.error("No [overlay_watchdog] section in config.ini.")
        sys.exit(1)

    c = cfg["overlay_watchdog"]

    enabled = c.getboolean("enable", fallback=True)
    if not enabled:
        logger.info("[overlay_watchdog] enable=false, exiting.")
        sys.exit(0)

    video_source = c.get("video_source", fallback="")
    if not video_source:
        logger.error("overlay_watchdog.video_source missing.")
        sys.exit(1)

    watchdog_zone_json = c.get("watchdog_zone_json", fallback="./data/watchdog_zone.json")
    poll_interval_sec = c.getfloat("poll_interval_sec", fallback=1.0)
    stuck_threshold_sec = c.getfloat("stuck_threshold_sec", fallback=20.0)
    loop_reset_threshold = c.getint("loop_reset_threshold", fallback=3)

    # Human layout, e.g. "DD/MM/YYYY  HH:MM:SS"
    layout_str = c.get("timestamp_format", fallback="DD/MM/YYYY HH:MM:SS")
    python_ts_format = layout_to_strftime(layout_str)

    enable_realtime_lag_check = c.getboolean("enable_realtime_lag_check", fallback=False)
    max_realtime_lag_seconds = c.getfloat("max_realtime_lag_seconds", fallback=120.0)

    tesseract_psm = c.get("tesseract_psm", fallback="7")
    tesseract_whitelist = c.get("tesseract_whitelist", fallback="0123456789/: ")

    on_stuck_cmd = c.get("on_stuck_cmd", fallback="").strip()
    on_loop_cmd = c.get("on_loop_cmd", fallback="").strip()
    on_error_cmd = c.get("on_error_cmd", fallback="").strip()

    return {
        "video_source": video_source,
        "watchdog_zone_json": watchdog_zone_json,
        "poll_interval_sec": poll_interval_sec,
        "stuck_threshold_sec": stuck_threshold_sec,
        "loop_reset_threshold": loop_reset_threshold,
        "timestamp_layout_raw": layout_str,
        "timestamp_format": python_ts_format,
        "enable_realtime_lag_check": enable_realtime_lag_check,
        "max_realtime_lag_seconds": max_realtime_lag_seconds,
        "tesseract_psm": tesseract_psm,
        "tesseract_whitelist": tesseract_whitelist,
        "on_stuck_cmd": on_stuck_cmd,
        "on_loop_cmd": on_loop_cmd,
        "on_error_cmd": on_error_cmd,
    }

def load_watchdog_roi(json_path):
    if not os.path.exists(json_path):
        logger.error(f"Watchdog zone JSON not found: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    zones = data.get("watchdog_zones", [])
    if not zones:
        logger.error(f"No 'watchdog_zones' in {json_path}")
        sys.exit(1)

    z = zones[0]
    logger.info(f"Using watchdog zone: {z}")
    return z["x1"], z["y1"], z["x2"], z["y2"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def run_command(cmd, reason):
    if not cmd:
        logger.info(f"No command configured for {reason}. Skipping.")
        return

    if DRY_RUN_MODE:
        logger.warning(f"[DRY RUN] Would execute command for {reason}: {cmd}")
        return

    logger.warning(f"Executing command for {reason}: {cmd}")
    try:
        env = os.environ.copy()
        env["WD_REASON"] = reason
        subprocess.run(cmd, shell=True, env=env, check=False)
    except Exception as e:
        logger.error(f"Error running command '{cmd}': {e}")

def clean_ocr_text(text: str) -> str:
    # Strip + collapse spaces
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    # Compact version to detect date+time glued together
    compact = re.sub(r"\s+", "", text)

    # Handle cases like "14/11/202518:55:02" => "14/11/2025 18:55:02"
    m = re.match(r"(\d{2}/\d{2}/\d{4})(\d{2}:\d{2}:\d{2})$", compact)
    if m:
        return f"{m.group(1)} {m.group(2)}"

    return text

def ocr_timestamp(roi_bgr, timestamp_format, tess_psm, tess_whitelist):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    tess_cfg = f"--psm {tess_psm} -c tessedit_char_whitelist={tess_whitelist}"

    raw = pytesseract.image_to_string(bw, config=tess_cfg)
    cleaned = clean_ocr_text(raw)

    # Always show what Tesseract actually saw
    logger.info(f"OCR raw='{raw.strip()}' cleaned='{cleaned}'")

    if not cleaned:
        return None

    try:
        # Normalize spaces so 1 vs 2 spaces don't matter
        norm_text = re.sub(r"\s+", " ", cleaned)
        norm_fmt = re.sub(r"\s+", " ", timestamp_format)
        dt = datetime.strptime(norm_text, norm_fmt)
        return dt
    except Exception as e:
        logger.warning(f"Failed to parse '{cleaned}' with format '{timestamp_format}': {e}")
        return None

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main():
    cfg = load_overlay_config()
    x1, y1, x2, y2 = load_watchdog_roi(cfg["watchdog_zone_json"])

    video_source = cfg["video_source"]
    poll_interval = cfg["poll_interval_sec"]
    stuck_threshold = cfg["stuck_threshold_sec"]
    loop_reset_threshold = cfg["loop_reset_threshold"]
    layout_raw = cfg["timestamp_layout_raw"]
    timestamp_format = cfg["timestamp_format"]
    enable_realtime_lag = cfg["enable_realtime_lag_check"]
    max_lag = cfg["max_realtime_lag_seconds"]
    tess_psm = cfg["tesseract_psm"]
    tess_whitelist = cfg["tesseract_whitelist"]
    on_stuck_cmd = cfg["on_stuck_cmd"]
    on_loop_cmd = cfg["on_loop_cmd"]
    on_error_cmd = cfg["on_error_cmd"]

    logger.info("Starting overlay OCR watchdog.")
    if DRY_RUN_MODE:
        logger.info("DRY_RUN_MODE is ENABLED – no commands will actually be executed.")
    logger.info(f"Video source: {video_source}")
    logger.info(f"ROI: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    logger.info(f"Timestamp layout (config): {layout_raw}")
    logger.info(f"Timestamp format (strptime): {timestamp_format}")
    if enable_realtime_lag:
        logger.info(f"Realtime lag check enabled, max_lag={max_lag}s")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {video_source}")
        run_command(on_error_cmd, "open_failed")
        sys.exit(1)

    last_ts = None
    last_ts_change_mono = time.monotonic()
    backward_resets = 0

    try:
        while True:
            ok, frame = cap.read()
            now_mono = time.monotonic()
            now_wall = datetime.now()

            if not ok or frame is None:
                logger.error("Failed to read frame from stream.")
                run_command(on_error_cmd, "read_failed")
                time.sleep(poll_interval)
                continue

            h, w = frame.shape[:2]
            x1c = max(0, min(w - 1, x1))
            x2c = max(0, min(w,     x2))
            y1c = max(0, min(h - 1, y1))
            y2c = max(0, min(h,     y2))

            roi = frame[y1c:y2c, x1c:x2c]
            if roi.size == 0:
                logger.error("ROI is empty after clamping. Check watchdog_zone.json or stream resolution.")
                run_command(on_error_cmd, "empty_roi")
                time.sleep(poll_interval)
                continue

            current_ts = ocr_timestamp(roi, timestamp_format, tess_psm, tess_whitelist)
            if current_ts is None:
                time.sleep(poll_interval)
                continue

            logger.info(f"Overlay time: {current_ts.strftime('%Y-%m-%d %H:%M:%S')}")

            # --- deadloop / frozen time detection ---
            if last_ts is None:
                last_ts = current_ts
                last_ts_change_mono = now_mono
            else:
                if current_ts > last_ts:
                    # forward progress
                    last_ts = current_ts
                    last_ts_change_mono = now_mono
                    backward_resets = 0
                elif current_ts < last_ts:
                    # time jumped backwards; loop?
                    backward_resets += 1
                    logger.warning(
                        f"Overlay time went backwards ({backward_resets}/{loop_reset_threshold}): "
                        f"{current_ts} < {last_ts}"
                    )
                    if backward_resets >= loop_reset_threshold:
                        run_command(on_loop_cmd, "deadloop")
                        backward_resets = 0
                    # do not update last_ts/last_ts_change_mono
                else:
                    # same as last time; handled by "stuck" logic
                    pass

            # Stuck detection: overlay hasn't changed for stuck_threshold
            age = now_mono - last_ts_change_mono
            if age >= stuck_threshold:
                logger.error(f"Overlay timestamp stuck for {age:.1f}s (>= {stuck_threshold}).")
                run_command(on_stuck_cmd, "frozen_overlay")
                last_ts_change_mono = now_mono

            # Optional: realtime lag detection
            if enable_realtime_lag:
                lag = (now_wall - current_ts).total_seconds()
                if lag > max_lag:
                    logger.error(f"Overlay time lagging behind real time by {lag:.1f}s (> {max_lag}).")
                    run_command(on_stuck_cmd, "lagging_overlay")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Interrupted by user, exiting.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
