#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay OCR watchdog (Final Merged + Patched Version):

- Crops the "watchdog" ROI defined by region_masker (mode 3).
- OCRs the timestamp overlay with Tesseract.
- Detects Frozen, Deadloop, and Hard Lag.
- Features "Smart Deadloop": won't restart on minor backward jitter if lag is fine.
- Features "Brain Wipe": resets state after restart to prevent loops.
- Features "Suicide Prevention": alerts BEFORE killing the stream.
- Patched stuck detection: backward jitter also counts as "change" for freeze detection,
  so you don't get bogus "Frozen overlay" while the timestamp is actually moving.
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
from datetime import datetime, timedelta

# ---------------------------------------------------------------------
# Global switches
# ---------------------------------------------------------------------
DRY_RUN_MODE = False
# DRY_RUN_MODE = True

# ---------------------------------------------------------------------
# Sanity limits for overlay timestamps
# ---------------------------------------------------------------------
MAX_ABS_OVERLAY_SKEW_SEC = 365 * 24 * 3600  # 1 year

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
        "python-telegram-bot is not installed; Telegram alerts will be DISABLED."
    )


def _send_telegram_alert(message: str):
    if telegram is None:
        logger.error("Telegram library not available, cannot send alert.")
        return

    async def send_async():
        bot_token = os.getenv("DVR_YOLOV8_TELEGRAM_BOT_TOKEN")
        user_ids_str = os.getenv("DVR_YOLOV8_ALLOWED_TELEGRAM_USERS")

        if not bot_token or not user_ids_str:
            logger.error("Cannot send Telegram alert: Env vars missing.")
            return

        bot = telegram.Bot(token=bot_token)
        user_ids = [int(uid.strip()) for uid in user_ids_str.split(",") if uid.strip()]

        for user_id in user_ids:
            try:
                await bot.send_message(
                    chat_id=user_id, text=message, parse_mode="HTML"
                )
            except Exception as e:
                logger.error(f"Failed to send Telegram message to user {user_id}: {e}")

    try:
        asyncio.run(send_async())
        logger.info("Overlay watchdog alert sent.")
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
    if not cfg.get("send_telegram_alerts", False):
        return
    if reason.endswith("_recovered") and not cfg.get("send_recovery_alerts", True):
        return
    if reason == "lag_warning" and not cfg.get("lag_warn_send_telegram", True):
        return

    interval = cfg.get("telegram_repeat_interval_sec", 60.0)
    now_mono = time.monotonic()
    last = last_alert_times.get(reason)

    if last is not None and interval > 0 and (now_mono - last) < interval:
        return

    overlay_str = overlay_ts.strftime("%Y-%m-%d %H:%M:%S") if overlay_ts else "unknown"
    now_str = now_wall.strftime("%Y-%m-%d %H:%M:%S") if now_wall else "unknown"
    lag_str = f"{lag_seconds:.1f}" if lag_seconds is not None else "N/A"

    msgs = {
        "frozen_overlay": "<b>Reason:</b> Frozen overlay timestamp\nStream may be frozen or replaying old content.",
        "lagging_overlay": "<b>Reason:</b> Overlay timestamp lagging behind realtime\nSource may be stuck, looping, or heavily delayed.",
        "lag_warning": "<b>Reason:</b> Overlay timestamp noticeably behind realtime\nWarning only.",
        "deadloop": "<b>Reason:</b> Overlay timestamp jumped backwards repeatedly\nUsually indicates a replaying buffer or encoder loop.",
        "empty_overlay": "<b>Reason:</b> Timestamp overlay missing or unreadable (empty OCR)\n",
        "unparseable_overlay": "<b>Reason:</b> Timestamp text repeatedly unparseable\n",
        "overlay_recovered": "<b>Status:</b> Overlay timestamp readable again\nOCR is back to normal.",
        "lagging_overlay_recovered": "<b>Status:</b> Overlay lag back within acceptable range\n",
        "post_restart_status": "<b>Status:</b> Post-restart status snapshot\nSystem is running.",
    }

    base_msg = msgs.get(reason, f"<b>Reason:</b> {reason}")

    msg = (
        f"⚠️ <b>YOLO-DVR OCR WATCHDOG</b> ⚠️\n\n"
        f"{base_msg}\n"
        f"<b>Overlay time:</b> {overlay_str}\n"
        f"<b>System time:</b>  {now_str}\n"
        f"<b>Difference:</b> {lag_str} s\n"
    )

    # Beautify for recovered / info states
    if reason.endswith("_recovered") or reason == "post_restart_status":
        msg = msg.replace("⚠️", "✅")

    if restart_requested:
        msg += "\n\n<b>Action:</b> Restart requested for the stream pipeline."
        if DRY_RUN_MODE:
            msg += "\n<b>Note:</b> DRY_RUN_MODE is ON."

    _send_telegram_alert(msg)
    last_alert_times[reason] = now_mono


def layout_to_strftime(layout: str) -> str:
    s = layout
    s = s.replace("YYYY", "%Y").replace("YY", "%y").replace("DD", "%d")
    if "MM" in s:
        s = s.replace("MM", "%m", 1)
    s = s.replace("MM", "%M").replace("HH", "%H").replace("hh", "%I")
    s = s.replace("SS", "%S").replace("ss", "%S")
    return s


def load_overlay_config(path="config.ini"):
    cfg = configparser.ConfigParser(interpolation=None)
    if not cfg.read(path):
        logger.error("No config.ini found.")
        sys.exit(1)
    if not cfg.has_section("overlay_watchdog"):
        logger.error("No [overlay_watchdog] section.")
        sys.exit(1)

    c = cfg["overlay_watchdog"]
    if not c.getboolean("enable", fallback=True):
        logger.info("Disabled in config.")
        sys.exit(0)

    video_source = c.get("video_source", fallback="").strip()
    if not video_source:
        logger.error("overlay_watchdog.video_source is missing or empty.")
        sys.exit(1)

    layout_raw = c.get("timestamp_format", fallback="DD/MM/YYYY HH:MM:SS")

    cfg_dict = {
        "video_source": video_source,
        "watchdog_zone_json": c.get(
            "watchdog_zone_json", fallback="./data/watchdog_zone.json"
        ),
        "poll_interval_sec": c.getfloat("poll_interval_sec", fallback=1.0),
        "stuck_threshold_sec": c.getfloat("stuck_threshold_sec", fallback=20.0),
        "loop_reset_threshold": c.getint("loop_reset_threshold", fallback=3),
        "timestamp_layout_raw": layout_raw,
        "timestamp_format": layout_to_strftime(layout_raw),
        "enable_realtime_lag_check": c.getboolean(
            "enable_realtime_lag_check", fallback=False
        ),
        "max_realtime_lag_seconds": c.getfloat(
            "max_realtime_lag_seconds", fallback=120.0
        ),
        "restart_cooldown_sec": c.getfloat(
            "restart_cooldown_sec", fallback=30.0
        ),
        "lag_warn_enable": c.getboolean("lag_warn_enable", fallback=True),
        "lag_warn_threshold_seconds": c.getfloat(
            "lag_warn_threshold_seconds", fallback=20.0
        ),
        "lag_warn_send_telegram": c.getboolean(
            "lag_warn_send_telegram", fallback=True
        ),
        "tesseract_psm": c.get("tesseract_psm", fallback="7"),
        "tesseract_whitelist": c.get(
            "tesseract_whitelist", fallback="0123456789/: "
        ),
        "use_env_commands": c.getboolean("use_env_commands", fallback=True),
        "env_command_prefix": c.get("env_command_prefix", fallback="DVR_OVERLAY_"),
        "on_stuck_cmd": c.get("on_stuck_cmd", fallback="").strip(),
        "on_loop_cmd": c.get("on_loop_cmd", fallback="").strip(),
        "on_error_cmd": c.get("on_error_cmd", fallback="").strip(),
        "send_telegram_alerts": c.getboolean(
            "send_telegram_alerts", fallback=False
        ),
        "telegram_repeat_interval_sec": c.getfloat(
            "telegram_repeat_interval_sec", fallback=60.0
        ),
        "send_startup_telegram": c.getboolean(
            "send_startup_telegram", fallback=True
        ),
        "ocr_fail_threshold_count": c.getint(
            "ocr_fail_threshold_count", fallback=3
        ),
        "send_recovery_alerts": c.getboolean(
            "send_recovery_alerts", fallback=True
        ),
        "recovery_lag_threshold_seconds": c.getfloat(
            "recovery_lag_threshold_seconds", fallback=5.0
        ),
        "send_post_restart_status": c.getboolean(
            "send_post_restart_status", fallback=True
        ),
        "post_restart_status_delay_seconds": c.getfloat(
            "post_restart_status_delay_seconds", fallback=30.0
        ),
    }

    return cfg_dict


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
    return z["x1"], z["y1"], z["x2"], z["y2"]


def run_command(cmd, reason):
    if not cmd:
        return
    if DRY_RUN_MODE:
        logger.warning(f"[DRY RUN] Command for {reason}: {cmd}")
        return
    logger.warning(f"Executing command for {reason}: {cmd}")
    env = os.environ.copy()
    env["WD_REASON"] = reason
    env["WD_SOURCE"] = "overlay_ocr"
    subprocess.run(cmd, shell=True, env=env, check=False)


def clean_ocr_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    compact = re.sub(r"\s+", "", text)
    m = re.match(r"(\d{2}/\d{2}/\d{4})(\d{2}:\d{2}:\d{2})$", compact)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return text


def ocr_timestamp(roi_bgr, timestamp_format, tess_psm, tess_whitelist):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    _, bw = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    raw = pytesseract.image_to_string(
        bw,
        config=f"--psm {tess_psm} -c tessedit_char_whitelist={tess_whitelist}",
    )
    cleaned = clean_ocr_text(raw)
    if not cleaned:
        return None, cleaned
    try:
        norm_text = re.sub(r"\s+", " ", cleaned)
        norm_fmt = re.sub(r"\s+", " ", timestamp_format)
        dt = datetime.strptime(norm_text, norm_fmt)
        if abs((dt - datetime.now()).total_seconds()) > MAX_ABS_OVERLAY_SKEW_SEC:
            return None, cleaned
        return dt, cleaned
    except Exception:
        return None, cleaned


def grab_one_frame(video_source, on_error_cmd):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Cannot open {video_source}")
        run_command(on_error_cmd, "open_failed")
        cap.release()
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        logger.error("Failed to read frame")
        run_command(on_error_cmd, "read_failed")
        return None
    return frame


def send_startup_telegram_if_enabled(cfg, video_source, roi, on_stuck_cmd, on_loop_cmd, on_error_cmd):
    if not cfg.get("send_telegram_alerts") or not cfg.get("send_startup_telegram"):
        return
    x1, y1, x2, y2 = roi
    msg = (
        "ℹ️ <b>YOLO-DVR OCR WATCHDOG</b> ℹ️\n\n"
        "<b>Status:</b> Overlay watchdog started\n"
        f"<b>Local time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"<b>Video source:</b> {video_source}\n"
        f"<b>ROI:</b> x1={x1}, y1={y1}, x2={x2}, y2={y2}\n"
        f"<b>DRY_RUN_MODE:</b> {'ON' if DRY_RUN_MODE else 'OFF'}"
    )
    _send_telegram_alert(msg)


def main():
    cfg = load_overlay_config()

    # Resolve commands from environment if requested
    if cfg["use_env_commands"]:
        prefix = cfg["env_command_prefix"]
        cfg["on_stuck_cmd"] = os.getenv(f"{prefix}ON_STUCK_CMD", "").strip()
        cfg["on_loop_cmd"] = os.getenv(f"{prefix}ON_LOOP_CMD", "").strip()
        cfg["on_error_cmd"] = os.getenv(f"{prefix}ON_ERROR_CMD", "").strip()

    video_source = cfg["video_source"]
    x1, y1, x2, y2 = load_watchdog_roi(cfg["watchdog_zone_json"])

    logger.info("Starting overlay OCR watchdog.")
    logger.info(f"Video source: {video_source}")
    logger.info(f"ROI: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    logger.info(
        f"Timestamp layout (config): {cfg['timestamp_layout_raw']} -> {cfg['timestamp_format']}"
    )
    if cfg["enable_realtime_lag_check"]:
        logger.info(
            f"Realtime lag check enabled, max_lag={cfg['max_realtime_lag_seconds']}s"
        )
    if cfg["lag_warn_enable"]:
        logger.info(
            f"Soft lag warning enabled, threshold={cfg['lag_warn_threshold_seconds']:.1f}s"
        )
    if cfg["restart_cooldown_sec"] > 0:
        logger.info(f"Restart cooldown: {cfg['restart_cooldown_sec']:.1f}s")
    else:
        logger.info("Restart cooldown disabled (restart_cooldown_sec <= 0)")
    logger.info(f"OCR fail threshold count: {cfg['ocr_fail_threshold_count']}")
    if DRY_RUN_MODE:
        logger.info("DRY_RUN_MODE is ENABLED – no commands will actually be executed.")

    send_startup_telegram_if_enabled(
        cfg,
        video_source,
        (x1, y1, x2, y2),
        cfg["on_stuck_cmd"],
        cfg["on_loop_cmd"],
        cfg["on_error_cmd"],
    )

    if not cfg["on_stuck_cmd"] and not cfg["on_loop_cmd"] and not cfg["on_error_cmd"]:
        logger.warning(
            "No overlay watchdog commands configured "
            "(on_stuck_cmd/on_loop_cmd/on_error_cmd all empty). "
            "Watchdog will NOT run any shell actions; only logging/Telegram."
        )

    last_ts = None
    last_ts_change_mono = time.monotonic()
    backward_resets = 0
    last_alert_times: dict[str, float] = {}
    bad_ocr_count = 0
    last_restart_mono = None
    in_hard_lag = False
    pending_post_restart_status = False
    post_restart_status_due_mono = None

    def reset_state_after_restart(now_mono: float):
        nonlocal last_ts, last_ts_change_mono, backward_resets, bad_ocr_count, in_hard_lag
        last_ts = None
        last_ts_change_mono = now_mono
        backward_resets = 0
        bad_ocr_count = 0
        in_hard_lag = False

    def perform_restart_sequence(reason, cmd, now_mono, current_ts, now_wall, lag):
        nonlocal last_restart_mono, pending_post_restart_status, post_restart_status_due_mono

        maybe_send_overlay_alert(
            reason,
            current_ts,
            now_wall,
            lag,
            cfg,
            last_alert_times,
            restart_requested=True,
        )

        run_command(cmd, reason)

        last_restart_mono = now_mono
        reset_state_after_restart(now_mono)

        if cfg["send_post_restart_status"] and cfg["post_restart_status_delay_seconds"] > 0:
            pending_post_restart_status = True
            post_restart_status_due_mono = (
                now_mono + cfg["post_restart_status_delay_seconds"]
            )

    try:
        while True:
            now_mono = time.monotonic()

            # 1. Grab frame first (avoid counting connection time into lag)
            frame = grab_one_frame(video_source, cfg["on_error_cmd"])

            # 2. Take wall clock time immediately after frame
            now_wall = datetime.now()

            if frame is None:
                time.sleep(cfg["poll_interval_sec"])
                continue

            h, w = frame.shape[:2]
            x1c = max(0, min(w - 1, x1))
            x2c = max(0, min(w, x2))
            y1c = max(0, min(h - 1, y1))
            y2c = max(0, min(h, y2))

            roi = frame[y1c:y2c, x1c:x2c]
            if roi.size == 0:
                logger.error("Empty ROI after clamping; check watchdog_zone.json.")
                run_command(cfg["on_error_cmd"], "empty_roi")
                maybe_send_overlay_alert(
                    "empty_overlay",
                    None,
                    now_wall,
                    None,
                    cfg,
                    last_alert_times,
                    restart_requested=False,
                )
                time.sleep(cfg["poll_interval_sec"])
                continue

            current_ts, cleaned_text = ocr_timestamp(
                roi,
                cfg["timestamp_format"],
                cfg["tesseract_psm"],
                cfg["tesseract_whitelist"],
            )

            if not cleaned_text or current_ts is None:
                bad_ocr_count += 1
                reason = "empty_overlay" if not cleaned_text else "unparseable_overlay"
                logger.warning(
                    f"OCR failure type={reason} (consecutive={bad_ocr_count})"
                )
                if bad_ocr_count >= cfg["ocr_fail_threshold_count"]:
                    maybe_send_overlay_alert(
                        reason,
                        None,
                        now_wall,
                        None,
                        cfg,
                        last_alert_times,
                        restart_requested=False,
                    )
                time.sleep(cfg["poll_interval_sec"])
                continue

            # We have a valid timestamp
            lag = (now_wall - current_ts).total_seconds()

            if bad_ocr_count >= cfg["ocr_fail_threshold_count"]:
                logger.info("OCR recovered: overlay timestamp readable again.")
                maybe_send_overlay_alert(
                    "overlay_recovered",
                    current_ts,
                    now_wall,
                    lag,
                    cfg,
                    last_alert_times,
                    restart_requested=False,
                )
            bad_ocr_count = 0

            logger.info(
                f"Overlay time: {current_ts.strftime('%Y-%m-%d %H:%M:%S')} (diff {lag:.1f}s)"
            )

            # Soft lag warning (purely informational)
            if cfg["lag_warn_enable"] and lag > cfg["lag_warn_threshold_seconds"]:
                maybe_send_overlay_alert(
                    "lag_warning",
                    current_ts,
                    now_wall,
                    lag,
                    cfg,
                    last_alert_times,
                    restart_requested=False,
                )

            # --------------------------------------------------------------
            # DEADLOOP / ORDERING LOGIC
            # --------------------------------------------------------------
            if last_ts is None:
                last_ts = current_ts
                last_ts_change_mono = now_mono
                backward_resets = 0

            elif current_ts > last_ts:
                # Forward progress: good
                last_ts = current_ts
                last_ts_change_mono = now_mono
                backward_resets = 0

            elif current_ts < last_ts:
                # Time went backwards – could be jitter or a real loop.
                backward_resets += 1
                # IMPORTANT: still treat this as "change" for stuck detection,
                # so that freeze detection does not trip while the overlay is moving.
                last_ts_change_mono = now_mono

                logger.warning(
                    f"Overlay time went backwards ({backward_resets}/{cfg['loop_reset_threshold']}): "
                    f"{current_ts} < {last_ts}"
                )

                if backward_resets >= cfg["loop_reset_threshold"]:
                    should_restart = True

                    # Smart deadloop: only restart if we're also badly lagging
                    if (
                        cfg["enable_realtime_lag_check"]
                        and cfg["max_realtime_lag_seconds"] > 0
                        and lag <= cfg["max_realtime_lag_seconds"]
                    ):
                        should_restart = False
                        logger.info(
                            "Deadloop detected but ignored (overlay within realtime bounds)."
                        )
                        maybe_send_overlay_alert(
                            "deadloop",
                            current_ts,
                            now_wall,
                            lag,
                            cfg,
                            last_alert_times,
                            restart_requested=False,
                        )
                        backward_resets = 0

                    if should_restart and cfg["on_loop_cmd"]:
                        elapsed = (
                            None
                            if last_restart_mono is None
                            else now_mono - last_restart_mono
                        )
                        if (
                            cfg["restart_cooldown_sec"] <= 0
                            or last_restart_mono is None
                            or elapsed >= cfg["restart_cooldown_sec"]
                        ):
                            perform_restart_sequence(
                                "deadloop",
                                cfg["on_loop_cmd"],
                                now_mono,
                                current_ts,
                                now_wall,
                                lag,
                            )
                        else:
                            logger.warning("Deadloop restart suppressed (cooldown).")
                            backward_resets = 0

            # Equal timestamp: handled purely by stuck detection below.

            # --------------------------------------------------------------
            # STUCK DETECTION (no change for stuck_threshold_sec)
            # --------------------------------------------------------------
            age = now_mono - last_ts_change_mono
            if age >= cfg["stuck_threshold_sec"]:
                logger.error(
                    f"Overlay timestamp stuck for {age:.1f}s "
                    f"(>= {cfg['stuck_threshold_sec']:.1f}s)."
                )
                if cfg["on_stuck_cmd"]:
                    elapsed = (
                        None
                        if last_restart_mono is None
                        else now_mono - last_restart_mono
                    )
                    if (
                        cfg["restart_cooldown_sec"] <= 0
                        or last_restart_mono is None
                        or elapsed >= cfg["restart_cooldown_sec"]
                    ):
                        perform_restart_sequence(
                            "frozen_overlay",
                            cfg["on_stuck_cmd"],
                            now_mono,
                            current_ts,
                            now_wall,
                            lag,
                        )
                    else:
                        logger.warning("Frozen restart suppressed (cooldown).")
                        maybe_send_overlay_alert(
                            "frozen_overlay",
                            current_ts,
                            now_wall,
                            lag,
                            cfg,
                            last_alert_times,
                            restart_requested=False,
                        )
                        # Reset age baseline so we don't spam on every loop
                        last_ts_change_mono = now_mono
                else:
                    maybe_send_overlay_alert(
                        "frozen_overlay",
                        current_ts,
                        now_wall,
                        lag,
                        cfg,
                        last_alert_times,
                        restart_requested=False,
                    )
                    last_ts_change_mono = now_mono

            # --------------------------------------------------------------
            # HARD REALTIME LAG + RECOVERY
            # --------------------------------------------------------------
            if cfg["enable_realtime_lag_check"] and cfg["max_realtime_lag_seconds"] > 0:
                if lag > cfg["max_realtime_lag_seconds"]:
                    logger.error(
                        f"Overlay time lagging behind realtime by {lag:.1f}s "
                        f"(> {cfg['max_realtime_lag_seconds']:.1f}s)."
                    )
                    if cfg["on_stuck_cmd"]:
                        elapsed = (
                            None
                            if last_restart_mono is None
                            else now_mono - last_restart_mono
                        )
                        if (
                            cfg["restart_cooldown_sec"] <= 0
                            or last_restart_mono is None
                            or elapsed >= cfg["restart_cooldown_sec"]
                        ):
                            perform_restart_sequence(
                                "lagging_overlay",
                                cfg["on_stuck_cmd"],
                                now_mono,
                                current_ts,
                                now_wall,
                                lag,
                            )
                        else:
                            logger.warning("Hard-lag restart suppressed (cooldown).")
                            maybe_send_overlay_alert(
                                "lagging_overlay",
                                current_ts,
                                now_wall,
                                lag,
                                cfg,
                                last_alert_times,
                                restart_requested=False,
                            )
                    else:
                        maybe_send_overlay_alert(
                            "lagging_overlay",
                            current_ts,
                            now_wall,
                            lag,
                            cfg,
                            last_alert_times,
                            restart_requested=False,
                        )
                    in_hard_lag = True

                else:
                    if in_hard_lag and cfg["send_recovery_alerts"]:
                        if 0 <= lag <= cfg["recovery_lag_threshold_seconds"]:
                            logger.info(
                                f"Overlay lag back within recovery threshold: "
                                f"{lag:.1f}s (<= {cfg['recovery_lag_threshold_seconds']:.1f}s)."
                            )
                            maybe_send_overlay_alert(
                                "lagging_overlay_recovered",
                                current_ts,
                                now_wall,
                                lag,
                                cfg,
                                last_alert_times,
                                restart_requested=False,
                            )
                            in_hard_lag = False

            # --------------------------------------------------------------
            # POST-RESTART STATUS SNAPSHOT
            # --------------------------------------------------------------
            if pending_post_restart_status and post_restart_status_due_mono is not None:
                if now_mono >= post_restart_status_due_mono:
                    maybe_send_overlay_alert(
                        "post_restart_status",
                        current_ts,
                        now_wall,
                        lag,
                        cfg,
                        last_alert_times,
                        restart_requested=False,
                    )
                    pending_post_restart_status = False
                    post_restart_status_due_mono = None

            time.sleep(cfg["poll_interval_sec"])

    except KeyboardInterrupt:
        logger.info("Exiting on KeyboardInterrupt.")
    except Exception as e:
        logger.exception("CRITICAL CRASH in overlay watchdog.")
        try:
            _send_telegram_alert(f"⚠️ <b>WATCHDOG CRASHED</b>\n{e}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
