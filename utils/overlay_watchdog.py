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
- Can send Telegram alerts and run shell commands on stuck/loop/error.

Requires:
    pip install opencv-python pytesseract python-telegram-bot
    tesseract-ocr installed system-wide.

Env vars for Telegram (same as main watchdog):
    DVR_YOLOV8_TELEGRAM_BOT_TOKEN
    DVR_YOLOV8_ALLOWED_TELEGRAM_USERS   (comma-separated user IDs)
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
import asyncio
from datetime import datetime

# ---------------------------------------------------------------------
# Global switches
# ---------------------------------------------------------------------
# Set this to False once you're happy with behavior
DRY_RUN_MODE = False
# DRY_RUN_MODE = True

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
# Telegram (optional)
# ---------------------------------------------------------------------
try:
    import telegram
except ImportError:
    telegram = None
    logger.warning(
        "python-telegram-bot is not installed; Telegram alerts will be DISABLED "
        "for overlay_watchdog."
    )

def _send_telegram_alert(message: str):
    """
    Fire-and-forget Telegram sender using same env vars as the main watchdog.
    """
    if telegram is None:
        logger.error("Telegram library not available, cannot send alert.")
        return

    async def send_async():
        bot_token = os.getenv("DVR_YOLOV8_TELEGRAM_BOT_TOKEN")
        user_ids_str = os.getenv("DVR_YOLOV8_ALLOWED_TELEGRAM_USERS")

        if not bot_token or not user_ids_str:
            logger.error(
                "Cannot send Telegram alert. "
                "DVR_YOLOV8_TELEGRAM_BOT_TOKEN or DVR_YOLOV8_ALLOWED_TELEGRAM_USERS not set."
            )
            return

        bot = telegram.Bot(token=bot_token)
        user_ids = [int(uid.strip()) for uid in user_ids_str.split(",") if uid.strip()]

        for user_id in user_ids:
            try:
                await bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.error(f"Failed to send Telegram message to user {user_id}: {e}")

    try:
        asyncio.run(send_async())
        logger.info("Overlay watchdog alert sent to all allowed Telegram users.")
    except Exception as e:
        logger.error(f"Error while sending Telegram alert: {e}")

def maybe_send_overlay_alert(
    reason: str,
    overlay_ts: datetime | None,
    now_wall: datetime | None,
    lag_seconds: float | None,
    cfg: dict,
    last_alert_times: dict,
    restart_requested: bool = False,
):

    """
    Rate-limited Telegram alerts per 'reason'.
    """
    if not cfg.get("send_telegram_alerts", False):
        return

    # Optional separate switch for soft lag warnings
    if reason == "lag_warning" and not cfg.get("lag_warn_send_telegram", True):
        return

    interval = cfg.get("telegram_repeat_interval_sec", 60.0)
    now_mono = time.monotonic()
    last = last_alert_times.get(reason)

    if last is not None and interval > 0 and (now_mono - last) < interval:
        # Too soon since last alert of this type
        return

    overlay_str = overlay_ts.strftime("%Y-%m-%d %H:%M:%S") if overlay_ts else "unknown"
    now_str = now_wall.strftime("%Y-%m-%d %H:%M:%S") if now_wall else "unknown"
    lag_str = f"{lag_seconds:.1f}" if lag_seconds is not None else "N/A"

    if reason == "frozen_overlay":
        threshold = cfg.get("stuck_threshold_sec", 0.0)
        msg = (
            "⚠️ <b>YOLO-DVR OCR WATCHDOG</b> ⚠️\n\n"
            "<b>Reason:</b> Frozen overlay timestamp\n"
            f"<b>Overlay time:</b> {overlay_str}\n"
            f"<b>System time:</b>  {now_str}\n"
            f"<b>Difference:</b> {lag_str} s\n"
            f"<b>Stuck threshold:</b> {threshold:.1f} s\n\n"
            "The on-screen timestamp has not advanced as expected.\n"
            "Stream may be frozen or replaying old content."
        )
    elif reason == "lagging_overlay":
        threshold = cfg.get("max_realtime_lag_seconds", 0.0)
        msg = (
            "⚠️ <b>YOLO-DVR OCR WATCHDOG</b> ⚠️\n\n"
            "<b>Reason:</b> Overlay timestamp lagging behind realtime\n"
            f"<b>Overlay time:</b> {overlay_str}\n"
            f"<b>System time:</b>  {now_str}\n"
            f"<b>Difference:</b> {lag_str} s\n"
            f"<b>Lag threshold:</b> {threshold:.1f} s\n\n"
            "The overlay time is significantly behind realtime.\n"
            "Source may be stuck, looping, or heavily delayed."
        )
    elif reason == "lag_warning":
        threshold = cfg.get("lag_warn_threshold_seconds", 0.0)
        msg = (
            "⚠️ <b>YOLO-DVR OCR WATCHDOG</b> ⚠️\n\n"
            "<b>Reason:</b> Overlay timestamp noticeably behind realtime\n"
            f"<b>Overlay time:</b> {overlay_str}\n"
            f"<b>System time:</b>  {now_str}\n"
            f"<b>Difference:</b> {lag_str} s\n"
            f"<b>Warning threshold:</b> {threshold:.1f} s\n\n"
            "Stream appears to be lagging significantly but has not yet crossed the "
            "hard restart threshold."
        )        
    elif reason == "deadloop":
        loop_thr = cfg.get("loop_reset_threshold", 0)
        msg = (
            "⚠️ <b>YOLO-DVR OCR WATCHDOG</b> ⚠️\n\n"
            "<b>Reason:</b> Overlay timestamp jumped backwards repeatedly\n"
            f"<b>Overlay time:</b> {overlay_str}\n"
            f"<b>System time:</b>  {now_str}\n"
            f"<b>Difference:</b> {lag_str} s\n"
            f"<b>Loop detection threshold:</b> {loop_thr} backward jumps\n\n"
            "The on-screen timestamp appears to be looping backwards.\n"
            "This usually indicates a replaying buffer or encoder loop."
        )
    elif reason == "empty_overlay":
        msg = (
            "⚠️ <b>YOLO-DVR OCR WATCHDOG</b> ⚠️\n\n"
            "<b>Reason:</b> Timestamp overlay missing or unreadable (empty OCR)\n"
            f"<b>Overlay time:</b> {overlay_str}\n"
            f"<b>System time:</b>  {now_str}\n\n"
            "OCR returned an empty result for the timestamp region repeatedly.\n"
            "Overlay may be hidden, obstructed, or changed format."
        )
    elif reason == "unparseable_overlay":
        msg = (
            "⚠️ <b>YOLO-DVR OCR WATCHDOG</b> ⚠️\n\n"
            "<b>Reason:</b> Timestamp text repeatedly unparseable\n"
            f"<b>Overlay time:</b> {overlay_str}\n"
            f"<b>System time:</b>  {now_str}\n\n"
            "OCR keeps returning garbage or text that does not match the expected "
            "timestamp layout.\n"
            "Overlay may be distorted, hidden behind graphics, or the format changed."
        )
    elif reason == "overlay_recovered":
        # "All clear" message when OCR has been bad and now returns to normal
        msg = (
            "✅ <b>YOLO-DVR OCR WATCHDOG</b> ✅\n\n"
            "<b>Status:</b> Overlay timestamp readable again\n"
            f"<b>Overlay time:</b> {overlay_str}\n"
            f"<b>System time:</b>  {now_str}\n"
            f"<b>Difference:</b> {lag_str} s\n\n"
            "OCR is once again returning a valid timestamp for the configured overlay region."
        )
    else:
        # Generic error / open_failed / read_failed / empty_roi
        msg = (
            "⚠️ <b>YOLO-DVR OCR WATCHDOG</b> ⚠️\n\n"
            f"<b>Reason:</b> {reason}\n"
            f"<b>Overlay time:</b> {overlay_str}\n"
            f"<b>System time:</b>  {now_str}\n"
            f"<b>Difference:</b> {lag_str} s\n\n"
            "An error was detected while reading or interpreting the overlay."
        )

    # If a restart/action was requested, annotate the message WITHOUT revealing any command
    if restart_requested:
        msg += "\n\n<b>Action:</b> Restart requested for the stream pipeline."
        if DRY_RUN_MODE:
            msg += "\n<b>Note:</b> DRY_RUN_MODE is ON, command was NOT actually executed."

    _send_telegram_alert(msg)
    last_alert_times[reason] = now_mono

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

    # Global restart cooldown (seconds) – 0 = disabled (no cooldown)
    restart_cooldown_sec = c.getfloat("restart_cooldown_sec", fallback=30.0)

    # Soft lag warning (no restart, just warns)
    lag_warn_enable = c.getboolean("lag_warn_enable", fallback=True)
    lag_warn_threshold_seconds = c.getfloat(
        "lag_warn_threshold_seconds", fallback=20.0
    )
    lag_warn_send_telegram = c.getboolean("lag_warn_send_telegram", fallback=True)

    tesseract_psm = c.get("tesseract_psm", fallback="7")
    tesseract_whitelist = c.get("tesseract_whitelist", fallback="0123456789/: ")

    # ---- command source: env vs config.ini ----
    use_env_commands = c.getboolean("use_env_commands", fallback=True)
    env_cmd_prefix = c.get("env_command_prefix", fallback="DVR_OVERLAY_")

    if use_env_commands:
        # Read from environment (e.g. loaded via .yolo_env / .env)
        on_stuck_cmd = os.getenv(f"{env_cmd_prefix}ON_STUCK_CMD", "").strip()
        on_loop_cmd  = os.getenv(f"{env_cmd_prefix}ON_LOOP_CMD", "").strip()
        on_error_cmd = os.getenv(f"{env_cmd_prefix}ON_ERROR_CMD", "").strip()
    else:
        # Read directly from config.ini
        on_stuck_cmd = c.get("on_stuck_cmd", fallback="").strip()
        on_loop_cmd  = c.get("on_loop_cmd", fallback="").strip()
        on_error_cmd = c.get("on_error_cmd", fallback="").strip()

    # Telegram settings
    send_telegram_alerts = c.getboolean("send_telegram_alerts", fallback=False)
    telegram_repeat_interval_sec = c.getfloat("telegram_repeat_interval_sec", fallback=60.0)
    send_startup_telegram = c.getboolean("send_startup_telegram", fallback=True)

    # OCR failure escalation
    ocr_fail_threshold_count = c.getint("ocr_fail_threshold_count", fallback=3)

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
        "send_telegram_alerts": send_telegram_alerts,
        "telegram_repeat_interval_sec": telegram_repeat_interval_sec,
        "ocr_fail_threshold_count": ocr_fail_threshold_count,
        "use_env_commands": use_env_commands,
        "env_command_prefix": env_cmd_prefix,
        "send_startup_telegram": send_startup_telegram,
        "lag_warn_enable": lag_warn_enable,
        "lag_warn_threshold_seconds": lag_warn_threshold_seconds,
        "lag_warn_send_telegram": lag_warn_send_telegram,
        "restart_cooldown_sec": restart_cooldown_sec,        
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
        env["WD_SOURCE"] = "overlay_ocr"
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
    """
    Returns (parsed_datetime_or_None, cleaned_text).
    cleaned_text is always the postprocessed OCR string (possibly empty).
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    tess_cfg = f"--psm {tess_psm} -c tessedit_char_whitelist={tess_whitelist}"

    raw = pytesseract.image_to_string(bw, config=tess_cfg)
    cleaned = clean_ocr_text(raw)

    if not cleaned:
        # Empty OCR is definitely suspicious
        logger.warning(f"OCR timestamp EMPTY. raw='{raw.strip()}' cleaned='{cleaned}'")
        return None, cleaned

    # Normal OCR log for analysis
    logger.info(f"OCR raw='{raw.strip()}' cleaned='{cleaned}'")

    try:
        # Normalize spaces so 1 vs 2 spaces don't matter
        norm_text = re.sub(r"\s+", " ", cleaned)
        norm_fmt = re.sub(r"\s+", " ", timestamp_format)
        dt = datetime.strptime(norm_text, norm_fmt)
        return dt, cleaned
    except Exception as e:
        logger.warning(f"Failed to parse '{cleaned}' with format '{timestamp_format}': {e}")
        return None, cleaned

def grab_one_frame(video_source, on_error_cmd):
    """
    Open stream, grab one frame, close immediately.
    This forces a fresh connection per poll.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {video_source}")
        run_command(on_error_cmd, "open_failed")
        cap.release()
        return None

    ok, frame = cap.read()
    if not ok or frame is None:
        logger.error("Failed to read frame from stream.")
        run_command(on_error_cmd, "read_failed")
        cap.release()
        return None

    cap.release()
    return frame

# startup notification on tg
def send_startup_telegram_if_enabled(
    cfg: dict,
    video_source: str,
    roi: tuple[int, int, int, int],
    on_stuck_cmd: str,
    on_loop_cmd: str,
    on_error_cmd: str,
):
    """
    Send a one-time 'watchdog started' info message, if enabled.
    """
    if not cfg.get("send_telegram_alerts", False):
        return
    if not cfg.get("send_startup_telegram", True):
        return

    x1, y1, x2, y2 = roi
    now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

    def _flag(name: str, cmd: str) -> str:
        return f"{name}={'configured' if cmd else 'none'}"

    # msg_lines = [
    #     "ℹ️ <b>YOLO-DVR OCR WATCHDOG</b> ℹ️",
    #     "",
    #     "<b>Status:</b> Overlay watchdog started",
    #     f"<b>Local time:</b> {now_local}",
    #     f"<b>Video source:</b> {video_source}",
    #     f"<b>ROI:</b> x1={x1}, y1={y1}, x2={x2}, y2={y2}",
    #     "",
    #     f"<b>DRY_RUN_MODE:</b> {'ON' if DRY_RUN_MODE else 'OFF'}",
    #     "<b>Commands:</b> "
    #     + ", ".join(
    #         [
    #             _flag("on_stuck_cmd", on_stuck_cmd),
    #             _flag("on_loop_cmd", on_loop_cmd),
    #             _flag("on_error_cmd", on_error_cmd),
    #         ]
    #     ),
    # ]

    msg_lines = [
        "ℹ️ <b>YOLO-DVR OCR WATCHDOG</b> ℹ️",
        "",
        "<b>Status:</b> Overlay watchdog started",
        f"<b>Local time:</b> {now_local}",
        "",
        f"<b>DRY_RUN_MODE:</b> {'ON' if DRY_RUN_MODE else 'OFF'}",
    ]

    _send_telegram_alert("\n".join(msg_lines))

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
    lag_warn_enable = cfg["lag_warn_enable"]
    lag_warn_threshold = cfg["lag_warn_threshold_seconds"]
    lag_warn_send_telegram = cfg["lag_warn_send_telegram"]
    restart_cooldown_sec = cfg["restart_cooldown_sec"]    
    tess_psm = cfg["tesseract_psm"]
    tess_whitelist = cfg["tesseract_whitelist"]
    on_stuck_cmd = cfg["on_stuck_cmd"]
    on_loop_cmd = cfg["on_loop_cmd"]
    on_error_cmd = cfg["on_error_cmd"]
    ocr_fail_threshold = cfg["ocr_fail_threshold_count"]

    logger.info("Starting overlay OCR watchdog.")
    if DRY_RUN_MODE:
        logger.info("DRY_RUN_MODE is ENABLED – no commands will actually be executed.")
    logger.info(f"Video source: {video_source}")
    logger.info(f"ROI: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    logger.info(f"Timestamp layout (config): {layout_raw}")
    logger.info(f"Timestamp format (strptime): {timestamp_format}")
    if enable_realtime_lag:
        logger.info(f"Realtime lag check enabled, max_lag={max_lag}s")
    if lag_warn_enable:
        logger.info(
            f"Soft lag warning enabled, threshold={lag_warn_threshold:.1f}s"
        )
    if restart_cooldown_sec > 0:
        logger.info(f"Restart cooldown: {restart_cooldown_sec:.1f}s")
    else:
        logger.info("Restart cooldown disabled (restart_cooldown_sec <= 0)")
    logger.info(f"OCR fail threshold count: {ocr_fail_threshold}")

    # send startup message on Telegram if enabled
    send_startup_telegram_if_enabled(
        cfg,
        video_source,
        (x1, y1, x2, y2),
        on_stuck_cmd,
        on_loop_cmd,
        on_error_cmd,
    )

    # Info about command source
    if cfg.get("use_env_commands", False):
        logger.info(
            "Overlay watchdog commands sourced from ENV "
            f"(prefix='{cfg.get('env_command_prefix')}')."
        )
    else:
        logger.info("Overlay watchdog commands sourced from config.ini.")

    # Warn if nothing is configured
    if not on_stuck_cmd and not on_loop_cmd and not on_error_cmd:
        logger.warning(
            "No overlay watchdog commands configured "
            "(on_stuck_cmd/on_loop_cmd/on_error_cmd all empty). "
            "Watchdog will NOT run any shell actions; only logging/Telegram."
        )
    else:
        if not on_stuck_cmd:
            logger.info(
                "on_stuck_cmd is empty – frozen overlay will NOT trigger a shell command."
            )
        if not on_loop_cmd:
            logger.info(
                "on_loop_cmd is empty – deadloop overlay will NOT trigger a shell command."
            )
        if not on_error_cmd:
            logger.info(
                "on_error_cmd is empty – open/read/ROI errors will NOT trigger a shell command."
            )

    last_ts = None
    last_ts_change_mono = time.monotonic()
    backward_resets = 0
    last_alert_times: dict[str, float] = {}
    bad_ocr_count = 0  # consecutive empty or unparseable OCR results
    last_restart_mono: float | None = None  # last time we actually issued a restart

    try:
        while True:
            now_mono = time.monotonic()
            now_wall = datetime.now()

            frame = grab_one_frame(video_source, on_error_cmd)
            if frame is None:
                # Couldn't get a frame; wait and retry
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
                maybe_send_overlay_alert(
                    "empty_roi", None, now_wall, None, cfg, last_alert_times
                )
                time.sleep(poll_interval)
                continue

            current_ts, cleaned_text = ocr_timestamp(
                roi, timestamp_format, tess_psm, tess_whitelist
            )

            # Handle OCR failures: empty or unparseable
            if not cleaned_text:
                bad_ocr_count += 1
                logger.warning(f"OCR empty ({bad_ocr_count} consecutive)")
                if bad_ocr_count >= ocr_fail_threshold:
                    maybe_send_overlay_alert(
                        "empty_overlay", None, now_wall, None, cfg, last_alert_times
                    )
                time.sleep(poll_interval)
                continue

            if current_ts is None:
                bad_ocr_count += 1
                logger.warning(f"OCR unparseable ({bad_ocr_count} consecutive)")
                if bad_ocr_count >= ocr_fail_threshold:
                    maybe_send_overlay_alert(
                        "unparseable_overlay", None, now_wall, None, cfg, last_alert_times
                    )
                time.sleep(poll_interval)
                continue

            # Got a valid timestamp: compute lag first
            lag = (now_wall - current_ts).total_seconds()

            # If we had a long run of bad OCR before this, send an "all clear"
            if bad_ocr_count >= ocr_fail_threshold:
                logger.info("OCR recovered: overlay timestamp readable again.")
                maybe_send_overlay_alert(
                    "overlay_recovered", current_ts, now_wall, lag, cfg, last_alert_times
                )

            # Reset bad OCR streak
            bad_ocr_count = 0

            # Nice human-readable lag info
            if abs(lag) < 0.5:
                lag_info = " (in sync)"
            elif lag > 0:
                lag_info = f" (lagging {lag:.1f} sec behind)"
            else:
                lag_info = f" (ahead {abs(lag):.1f} sec)"

            logger.info(
                f"Overlay time: {current_ts.strftime('%Y-%m-%d %H:%M:%S')}{lag_info}"
            )

            # Soft lag warning (no restart, just warning/TG)
            if lag > 0 and lag_warn_enable:
                send_soft_warning = False

                if enable_realtime_lag and max_lag > 0:
                    # Only send "lag_warning" if we are between soft and hard thresholds
                    if lag_warn_threshold < lag <= max_lag:
                        send_soft_warning = True
                else:
                    # No hard threshold – just use the soft one
                    if lag > lag_warn_threshold:
                        send_soft_warning = True

                if send_soft_warning:
                    logger.warning(
                        f"Overlay lag {lag:.1f}s exceeds soft warning threshold "
                        f"{lag_warn_threshold:.1f}s."
                    )
                    if lag_warn_send_telegram:
                        maybe_send_overlay_alert(
                            "lag_warning",
                            current_ts,
                            now_wall,
                            lag,
                            cfg,
                            last_alert_times,
                            restart_requested=False,
                        )

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
                        restart_requested = False
                        if on_loop_cmd:
                            elapsed = (
                                None if last_restart_mono is None
                                else now_mono - last_restart_mono
                            )
                            if (
                                restart_cooldown_sec <= 0
                                or last_restart_mono is None
                                or elapsed >= restart_cooldown_sec
                            ):
                                restart_requested = True
                                run_command(on_loop_cmd, "deadloop")
                                last_restart_mono = now_mono
                            else:
                                logger.warning(
                                    "Restart suppressed due to cooldown: "
                                    f"{elapsed:.1f}s since last restart "
                                    f"(< {restart_cooldown_sec:.1f}s)."
                                )

                        maybe_send_overlay_alert(
                            "deadloop",
                            current_ts,
                            now_wall,
                            lag,
                            cfg,
                            last_alert_times,
                            restart_requested=restart_requested,
                        )
                        backward_resets = 0
                    # do not update last_ts/last_ts_change_mono
                else:
                    # same as last time; handled by "stuck" logic
                    pass

            # Stuck detection: overlay hasn't changed for stuck_threshold
            age = now_mono - last_ts_change_mono
            if age >= stuck_threshold:
                logger.error(
                    f"Overlay timestamp stuck for {age:.1f}s (>= {stuck_threshold})."
                )
                restart_requested = False
                if on_stuck_cmd:
                    elapsed = (
                        None if last_restart_mono is None
                        else now_mono - last_restart_mono
                    )
                    if (
                        restart_cooldown_sec <= 0
                        or last_restart_mono is None
                        or elapsed >= restart_cooldown_sec
                    ):
                        restart_requested = True
                        run_command(on_stuck_cmd, "frozen_overlay")
                        last_restart_mono = now_mono
                    else:
                        logger.warning(
                            "Restart suppressed due to cooldown: "
                            f"{elapsed:.1f}s since last restart "
                            f"(< {restart_cooldown_sec:.1f}s)."
                        )

                maybe_send_overlay_alert(
                    "frozen_overlay",
                    current_ts,
                    now_wall,
                    lag,
                    cfg,
                    last_alert_times,
                    restart_requested=restart_requested,
                )
                last_ts_change_mono = now_mono

            # Optional: realtime lag detection
            if enable_realtime_lag:
                if lag > max_lag:
                    logger.error(
                        f"Overlay time lagging behind real time by {lag:.1f}s (> {max_lag})."
                    )
                    restart_requested = False
                    if on_stuck_cmd:
                        elapsed = (
                            None if last_restart_mono is None
                            else now_mono - last_restart_mono
                        )
                        if (
                            restart_cooldown_sec <= 0
                            or last_restart_mono is None
                            or elapsed >= restart_cooldown_sec
                        ):
                            restart_requested = True
                            run_command(on_stuck_cmd, "lagging_overlay")
                            last_restart_mono = now_mono
                        else:
                            logger.warning(
                                "Restart suppressed due to cooldown: "
                                f"{elapsed:.1f}s since last restart "
                                f"(< {restart_cooldown_sec:.1f}s)."
                            )

                    maybe_send_overlay_alert(
                        "lagging_overlay",
                        current_ts,
                        now_wall,
                        lag,
                        cfg,
                        last_alert_times,
                        restart_requested=restart_requested,
                    )

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Interrupted by user, exiting.")

if __name__ == "__main__":
    main()
