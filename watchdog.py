# watchdog.py
#
# v2: A completely standalone script to monitor an RTMP stream for freezes.
#     - Fixes the asyncio/Telegram bug.
#     - Adds a --debug mode to visually tune sensitivity.
# 
# See `config.ini` for `[watchdog]` section to configure.
#
# To run normally:  python3 watchdog.py
# To debug/tune:     python3 watchdog.py --debug
#

import cv2
import numpy as np
import time
import configparser
import logging
import sys
import os
import threading
import asyncio
import argparse

# It requires the python-telegram-bot library.
# Install it with: pip install python-telegram-bot==13.15
try:
    import telegram
except ImportError:
    print("Error: The 'python-telegram-bot' library is not installed.")
    print("Please install it using: pip install 'python-telegram-bot==13.15'")
    sys.exit(1)

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Watchdog] - %(message)s',
    stream=sys.stdout,
)

# --- CORRECTED, Self-Contained Telegram Function ---
def send_telegram_alert_sync():
    """
    Synchronous wrapper that correctly runs the async send_telegram_alert function.
    This fixes the 'coroutine was never awaited' bug.
    """
    try:
        asyncio.run(send_telegram_alert_async())
        logging.info("Successfully sent stream freeze alert to all allowed users.")
    except Exception as e:
        logging.error(f"An error occurred in the Telegram alert thread: {e}")

async def send_telegram_alert_async():
    """
    Initializes its own Telegram bot instance and sends a freeze alert asynchronously.
    """
    bot_token = os.getenv('DVR_YOLOV8_TELEGRAM_BOT_TOKEN')
    user_ids_str = os.getenv('DVR_YOLOV8_ALLOWED_TELEGRAM_USERS')

    if not bot_token or not user_ids_str:
        logging.error("Cannot send Telegram alert. Environment variables for the bot are not set.")
        return

    bot = telegram.Bot(token=bot_token)
    user_ids = [int(uid.strip()) for uid in user_ids_str.split(',')]
    
    freeze_message = (
        "🚨❄️ <b>STREAM FROZEN</b> ❄️🚨\n\n"
        "The watchdog has detected that the video stream has been frozen for a significant duration.\n"
        "Please check the source (e.g., OBS)."
    )

    for user_id in user_ids:
        try:
            await bot.send_message(chat_id=user_id, text=freeze_message, parse_mode="HTML")
        except Exception as e:
            logging.error(f"Failed to send message to user {user_id}: {e}")

# --- Main Application ---
try:
    config = configparser.ConfigParser(interpolation=None)
    config.read('config.ini')

    STREAM_URL = config.get('stream', 'stream_url', fallback='rtmp://127.0.0.1:1935/live/stream')
    ENABLE_FREEZE_DETECTOR = config.getboolean('watchdog', 'enable_freeze_detector', fallback=False)
    FREEZE_DURATION = config.getint('watchdog', 'freeze_duration', fallback=10)
    FREEZE_SENSITIVITY_THRESHOLD = config.getint('watchdog', 'freeze_sensitivity_threshold', fallback=500)
    SEND_TELEGRAM_ALERT_ON_FREEZE = config.getboolean('watchdog', 'send_telegram_alert_on_freeze', fallback=True)
    POLL_INTERVAL = config.getint('watchdog', 'poll_interval_seconds', fallback=1)
except Exception as e:
    logging.error(f"FATAL: Could not read config.ini. Error: {e}")
    sys.exit(1)

def run_watchdog(debug_mode=False):
    if not ENABLE_FREEZE_DETECTOR:
        logging.info("Freeze detector is disabled in config.ini. Exiting.")
        return

    logging.info("Starting standalone stream freeze watchdog...")
    if debug_mode:
        debug_path = "watchdog_debug"
        os.makedirs(debug_path, exist_ok=True)
        logging.warning(f"DEBUG MODE IS ON. Images will be saved to the '{debug_path}' folder.")

    last_frame_signature = None
    freeze_start_time = None
    alert_sent_for_current_freeze = False
    cap = None

    while True:
        try:
            if cap is None or not cap.isOpened():
                logging.info("Attempting to connect to video stream...")
                cap = cv2.VideoCapture(STREAM_URL)
                if not cap.isOpened():
                    logging.warning("Failed to connect. Retrying in 10 seconds...")
                    time.sleep(10)
                    continue
                logging.info("Successfully connected.")

            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab a frame. Reconnecting...")
                cap.release()
                cap = None
                time.sleep(5)
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_signature = cv2.resize(gray_frame, (128, 72), interpolation=cv2.INTER_AREA)

            if last_frame_signature is not None:
                diff_image = cv2.absdiff(last_frame_signature, current_signature)
                diff_sum = int(np.sum(diff_image))

                # Update log message to be more informative
                log_message = f"Frame difference: {diff_sum} (Threshold: {FREEZE_SENSITIVITY_THRESHOLD})"
                logging.info(log_message)

                if debug_mode:
                    cv2.imwrite(os.path.join(debug_path, "current.png"), current_signature)
                    cv2.imwrite(os.path.join(debug_path, "last.png"), last_frame_signature)
                    cv2.imwrite(os.path.join(debug_path, "diff.png"), diff_image)

                if diff_sum < FREEZE_SENSITIVITY_THRESHOLD:
                    if freeze_start_time is None:
                        freeze_start_time = time.time()
                    
                    if (time.time() - freeze_start_time) > FREEZE_DURATION and not alert_sent_for_current_freeze:
                        logging.critical(f"STREAM FROZEN! No significant change for {FREEZE_DURATION} seconds.")
                        if SEND_TELEGRAM_ALERT_ON_FREEZE:
                            threading.Thread(target=send_telegram_alert_sync).start()
                        alert_sent_for_current_freeze = True
                else:
                    if freeze_start_time is not None:
                        logging.info("Stream is active again.")
                    freeze_start_time = None
                    alert_sent_for_current_freeze = False
            
            last_frame_signature = current_signature
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            logging.info("Watchdog stopped by user.")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
            if cap: cap.release()
            cap = None
            time.sleep(10)

    if cap: cap.release()
    logging.info("Watchdog has shut down.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone watchdog for detecting frozen RTMP streams.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode to save frame comparisons as images.")
    args = parser.parse_args()
    
    run_watchdog(debug_mode=args.debug)

