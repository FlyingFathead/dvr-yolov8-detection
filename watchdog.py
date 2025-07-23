# watchdog.py
#
# v10 (Definitive): Fixes the NameError bug from the previous version.
#     Includes stale connection detection and all other features.
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
import subprocess

try:
    import telegram
except ImportError:
    print("Error: The 'python-telegram-bot' library is not installed.")
    print("Please install it using: pip install 'python-telegram-bot'")
    sys.exit(1)

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Watchdog] - %(message)s',
    stream=sys.stdout,
)

# --- Self-Contained Telegram Functions ---
def _send_telegram_alert(message):
    """Generic alert sender to avoid code duplication."""
    async def send_async():
        bot_token = os.getenv('DVR_YOLOV8_TELEGRAM_BOT_TOKEN')
        user_ids_str = os.getenv('DVR_YOLOV8_ALLOWED_TELEGRAM_USERS')
        if not bot_token or not user_ids_str:
            logging.error("Cannot send Telegram alert. Environment variables for the bot are not set.")
            return
        bot = telegram.Bot(token=bot_token)
        user_ids = [int(uid.strip()) for uid in user_ids_str.split(',')]
        for user_id in user_ids:
            try:
                await bot.send_message(chat_id=user_id, text=message, parse_mode="HTML")
            except Exception as e:
                logging.error(f"Failed to send message to user {user_id}: {e}")
    try:
        asyncio.run(send_async())
        logging.info("Successfully sent alert to all allowed users.")
    except Exception as e:
        logging.error(f"An error occurred in the Telegram alert thread: {e}")

def send_freeze_alert_sync():
    _send_telegram_alert(
        "🚨❄️ <b>STREAM FROZEN</b> ❄️🚨\n\n"
        "The watchdog is connected, but the video stream has been static for a significant duration.\n"
        "Please check the source (e.g., OBS)."
    )

def send_disconnect_alert_sync():
    _send_telegram_alert(
        "🚨‼️ <b>STREAM OFFLINE</b> ‼️🚨\n\n"
        "The watchdog cannot connect to the stream source.\n"
        "Please check your streaming software or network connection."
    )

def send_restarting_alert_sync(reason="frozen"):
    """Sends an alert announcing the hard reset attempt."""
    if reason == "stale":
        message = (
            "⚠️ <b>HARD RESET</b> ⚠️\n\n"
            "Stream connection is STALE and UNRELIABLE. Forcing a restart.\n"
            "Monitoring will continue."
        )
    elif reason == "offline":
        message = (
            "⚠️ <b>HARD RESET</b> ⚠️\n\n"
            "Stream is OFFLINE. Attempting to force-restart the source via command line.\n"
            "Monitoring will continue."
        )
    else: # Default to frozen
        message = (
            "⚠️ <b>HARD RESET</b> ⚠️\n\n"
            "Stream is FROZEN. Attempting to force-restart the source via command line.\n"
            "Monitoring will continue."
        )
    _send_telegram_alert(message)

# --- Main Application ---
try:
    config = configparser.ConfigParser(interpolation=None)
    config.read('config.ini')
    STREAM_URL = config.get('stream', 'stream_url', fallback='rtmp://127.0.0.1:1935/live/stream')
    
    POLL_INTERVAL = config.getint('watchdog', 'poll_interval_seconds', fallback=1)
    SEND_TELEGRAM_ALERT_ON_FREEZE = config.getboolean('watchdog', 'send_telegram_alert_on_freeze', fallback=True)
    # Freeze Detection
    ENABLE_FREEZE_DETECTOR = config.getboolean('watchdog', 'enable_freeze_detector', fallback=True)
    FREEZE_DURATION_SECONDS = config.getint('watchdog', 'freeze_duration_seconds', fallback=10)
    FREEZE_SENSITIVITY_THRESHOLD = config.getint('watchdog', 'freeze_sensitivity_threshold', fallback=250)
    ENABLE_REPEATED_ALERTS = config.getboolean('watchdog', 'enable_repeated_alerts', fallback=True)
    REPEAT_ALERT_SECONDS = config.getint('watchdog', 'repeat_alert_interval_seconds', fallback=60)
    # Disconnect Detection
    ENABLE_DISCONNECT_ALERTS = config.getboolean('watchdog', 'enable_disconnect_alerts', fallback=True)
    DISCONNECT_THRESHOLD_SECONDS = config.getint('watchdog', 'disconnect_alert_threshold_seconds', fallback=60)
    REPEAT_DISCONNECT_SECONDS = config.getint('watchdog', 'repeat_disconnect_alert_interval_seconds', fallback=60)
    # Hard Reset Settings
    ENABLE_HARD_RESET = config.getboolean('watchdog', 'enable_hard_reset', fallback=False)
    HARD_RESET_COMMAND = config.get('watchdog', 'hard_reset_command', fallback='killall obs')
    
    # --- THIS IS THE FIX ---
    # This line was missing, causing the NameError.
    STALE_CONNECTION_THRESHOLD_SECONDS = config.getint('watchdog', 'stale_connection_threshold_seconds', fallback=25)

except Exception as e:
    logging.error(f"FATAL: Could not read config.ini. Error: {e}")
    sys.exit(1)

def execute_hard_reset(reason="frozen"):
    """Announces and executes the hard reset command."""
    if SEND_TELEGRAM_ALERT_ON_FREEZE:
        send_restarting_alert_sync(reason=reason)
    
    logging.warning(f"Executing hard reset command due to {reason} stream: '{HARD_RESET_COMMAND}'")
    try:
        subprocess.run(HARD_RESET_COMMAND, shell=True, check=True)
        logging.info("Hard reset command executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Hard reset command failed with exit code {e.returncode}: {e}")
    except Exception as e:
        logging.error(f"Failed to execute hard reset command: {e}")

def run_watchdog():
    logging.info("Starting standalone stream watchdog...")
    if ENABLE_FREEZE_DETECTOR: logging.info(f"Freeze detection is ENABLED.")
    if ENABLE_DISCONNECT_ALERTS: logging.info(f"Disconnect detection is ENABLED.")
    if ENABLE_HARD_RESET: logging.warning(f"Hard reset is ENABLED. Command: '{HARD_RESET_COMMAND}'")
    
    cap = None
    last_frame_signature, freeze_start_time, last_freeze_alert_time = None, None, None
    connection_fail_start_time, last_disconnect_alert_time = None, None

    while True:
        try:
            # --- Stream Connection Logic ---
            if cap is None or not cap.isOpened():
                if ENABLE_DISCONNECT_ALERTS:
                    if connection_fail_start_time is None:
                        logging.warning("Failed to connect to stream. Starting disconnect timer...")
                        connection_fail_start_time = time.time()
                    if (time.time() - connection_fail_start_time) > DISCONNECT_THRESHOLD_SECONDS:
                        time_to_alert = last_disconnect_alert_time is None or \
                                       (time.time() - last_disconnect_alert_time) > REPEAT_DISCONNECT_SECONDS
                        if time_to_alert:
                            last_disconnect_alert_time = time.time()
                            if ENABLE_HARD_RESET:
                                threading.Thread(target=execute_hard_reset, kwargs={"reason": "offline"}).start()
                            else:
                                logging.critical("STREAM OFFLINE! Sending disconnect alert.")
                                if SEND_TELEGRAM_ALERT_ON_FREEZE:
                                    threading.Thread(target=send_disconnect_alert_sync).start()
                logging.info("Attempting to connect to video stream...")
                cap = cv2.VideoCapture(STREAM_URL)
                if not cap.isOpened():
                    time.sleep(10)
                    continue
            
            if connection_fail_start_time is not None:
                logging.info("Stream connection has been RESTORED.")
                connection_fail_start_time, last_disconnect_alert_time = None, None

            # --- Frame Reading & Freeze Detection ---
            ret, frame = cap.read()
            if not ret:
                logging.warning("Connected but failed to grab frame. Re-establishing...")
                cap.release(); cap = None
                time.sleep(5)
                continue

            if ENABLE_FREEZE_DETECTOR:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                current_signature = cv2.resize(gray, (128, 72), interpolation=cv2.INTER_AREA)
                if last_frame_signature is not None:
                    diff_sum = int(np.sum(cv2.absdiff(last_frame_signature, current_signature)))
                    log_message = f"Frame difference: {diff_sum} (Threshold: {FREEZE_SENSITIVITY_THRESHOLD})"
                    if freeze_start_time is not None:
                        frozen_for = int(time.time() - freeze_start_time)
                        log_message += f" | Frozen for: {frozen_for}s"
                    logging.info(log_message)

                    if diff_sum < FREEZE_SENSITIVITY_THRESHOLD:
                        if freeze_start_time is None: freeze_start_time = time.time()
                        
                        if (time.time() - freeze_start_time) > STALE_CONNECTION_THRESHOLD_SECONDS:
                            logging.critical(f"STALE CONNECTION! Stream has been in a frozen state for over {STALE_CONNECTION_THRESHOLD_SECONDS}s. Forcing hard reset.")
                            if ENABLE_HARD_RESET:
                                execute_hard_reset(reason="stale")
                                freeze_start_time, last_freeze_alert_time = None, None
                                cap.release(); cap = None
                                continue
                            else:
                                pass

                        if (time.time() - freeze_start_time) > FREEZE_DURATION_SECONDS:
                            time_to_alert = last_freeze_alert_time is None or \
                                           (ENABLE_REPEATED_ALERTS and (time.time() - last_freeze_alert_time) > REPEAT_ALERT_SECONDS)
                            if time_to_alert:
                                last_freeze_alert_time = time.time()
                                if ENABLE_HARD_RESET:
                                    threading.Thread(target=execute_hard_reset, kwargs={"reason": "frozen"}).start()
                                else:
                                    logging.critical(f"STREAM FROZEN! Sending freeze alert.")
                                    if SEND_TELEGRAM_ALERT_ON_FREEZE:
                                        threading.Thread(target=send_freeze_alert_sync).start()
                    else:
                        if freeze_start_time is not None: logging.info("Stream freeze condition resolved.")
                        freeze_start_time, last_freeze_alert_time = None, None
                last_frame_signature = current_signature
            
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            logging.info("Watchdog stopped by user.")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
            if cap: cap.release(); cap = None
            time.sleep(10)

    if cap: cap.release()
    logging.info("Watchdog has shut down.")

if __name__ == "__main__":
    run_watchdog()

# # watchdog.py
# #
# # v8 (Final): Adds the ability to automatically run a system command
# #     to hard-reset the stream source (e.g., kill OBS) upon freeze detection.
# #
# import cv2
# import numpy as np
# import time
# import configparser
# import logging
# import sys
# import os
# import threading
# import asyncio
# import argparse
# import subprocess

# try:
#     import telegram
# except ImportError:
#     print("Error: The 'python-telegram-bot' library is not installed.")
#     print("Please install it using: pip install 'python-telegram-bot==13.15'")
#     sys.exit(1)

# # --- Basic Logging Setup ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [Watchdog] - %(message)s',
#     stream=sys.stdout,
# )

# # --- Self-Contained Telegram Functions ---
# def _send_telegram_alert(message):
#     """Generic alert sender to avoid code duplication."""
#     async def send_async():
#         bot_token = os.getenv('DVR_YOLOV8_TELEGRAM_BOT_TOKEN')
#         user_ids_str = os.getenv('DVR_YOLOV8_ALLOWED_TELEGRAM_USERS')
#         if not bot_token or not user_ids_str:
#             logging.error("Cannot send Telegram alert. Environment variables for the bot are not set.")
#             return
#         bot = telegram.Bot(token=bot_token)
#         user_ids = [int(uid.strip()) for uid in user_ids_str.split(',')]
#         for user_id in user_ids:
#             try:
#                 await bot.send_message(chat_id=user_id, text=message, parse_mode="HTML")
#             except Exception as e:
#                 logging.error(f"Failed to send message to user {user_id}: {e}")
#     try:
#         asyncio.run(send_async())
#         logging.info("Successfully sent alert to all allowed users.")
#     except Exception as e:
#         logging.error(f"An error occurred in the Telegram alert thread: {e}")

# def send_freeze_alert_sync():
#     _send_telegram_alert(
#         "🚨❄️ <b>STREAM FROZEN</b> ❄️🚨\n\n"
#         "The watchdog is connected, but the video stream has been static for a significant duration.\n"
#         "Please check the source (e.g., OBS)."
#     )

# def send_disconnect_alert_sync():
#     _send_telegram_alert(
#         "🚨‼️ <b>STREAM OFFLINE</b> ‼️🚨\n\n"
#         "The watchdog cannot connect to the stream source.\n"
#         "Please check your streaming software or network connection."
#     )

# def send_restarting_alert_sync(reason="frozen"):
#     """Sends an alert announcing the hard reset attempt."""
#     if reason == "offline":
#         message = (
#             "⚠️ <b>HARD RESET</b> ⚠️\n\n"
#             "Stream is OFFLINE. Attempting to force-restart the source via command line.\n"
#             "Monitoring will continue."
#         )
#     else: # Default to frozen
#         message = (
#             "⚠️ <b>HARD RESET</b> ⚠️\n\n"
#             "Stream is FROZEN. Attempting to force-restart the source via command line.\n"
#             "Monitoring will continue."
#         )
#     _send_telegram_alert(message)

# # --- Main Application ---
# try:
#     config = configparser.ConfigParser(interpolation=None)
#     config.read('config.ini')
#     STREAM_URL = config.get('stream', 'stream_url', fallback='rtmp://127.0.0.1:1935/live/stream')
    
#     POLL_INTERVAL = config.getint('watchdog', 'poll_interval_seconds', fallback=1)
#     SEND_TELEGRAM_ALERT_ON_FREEZE = config.getboolean('watchdog', 'send_telegram_alert_on_freeze', fallback=True)
#     # Freeze Detection
#     ENABLE_FREEZE_DETECTOR = config.getboolean('watchdog', 'enable_freeze_detector', fallback=True)
#     FREEZE_DURATION_SECONDS = config.getint('watchdog', 'freeze_duration_seconds', fallback=10)
#     FREEZE_SENSITIVITY_THRESHOLD = config.getint('watchdog', 'freeze_sensitivity_threshold', fallback=250)
#     ENABLE_REPEATED_ALERTS = config.getboolean('watchdog', 'enable_repeated_alerts', fallback=True)
#     REPEAT_ALERT_SECONDS = config.getint('watchdog', 'repeat_alert_interval_seconds', fallback=60)
#     # Disconnect Detection
#     ENABLE_DISCONNECT_ALERTS = config.getboolean('watchdog', 'enable_disconnect_alerts', fallback=True)
#     DISCONNECT_THRESHOLD_SECONDS = config.getint('watchdog', 'disconnect_alert_threshold_seconds', fallback=60)
#     REPEAT_DISCONNECT_SECONDS = config.getint('watchdog', 'repeat_disconnect_alert_interval_seconds', fallback=60)
#     # Hard Reset Settings
#     ENABLE_HARD_RESET = config.getboolean('watchdog', 'enable_hard_reset', fallback=False)
#     HARD_RESET_COMMAND = config.get('watchdog', 'hard_reset_command', fallback='killall obs')
#     HARD_RESET_PRE_WARNING_SECONDS = config.getint('watchdog', 'hard_reset_pre_warning_seconds', fallback=5)

# except Exception as e:
#     logging.error(f"FATAL: Could not read config.ini. Error: {e}")
#     sys.exit(1)

# except Exception as e:
#     logging.error(f"FATAL: Could not read config.ini. Error: {e}")
#     sys.exit(1)

# def execute_hard_reset(reason="frozen"):
#     """Announces and executes the hard reset command."""
#     if SEND_TELEGRAM_ALERT_ON_FREEZE:
#         # Pass the reason to the alert function for a more specific message
#         send_restarting_alert_sync(reason=reason)
    
#     logging.warning(f"Executing hard reset command due to {reason} stream: '{HARD_RESET_COMMAND}'")
#     try:
#         # Using shell=True to properly execute commands like 'killall'
#         subprocess.run(HARD_RESET_COMMAND, shell=True, check=True)
#         logging.info("Hard reset command executed successfully.")
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Hard reset command failed with exit code {e.returncode}: {e}")
#     except Exception as e:
#         logging.error(f"Failed to execute hard reset command: {e}")

# def run_watchdog():
#     logging.info("Starting standalone stream watchdog...")
#     if ENABLE_FREEZE_DETECTOR: logging.info(f"Freeze detection is ENABLED.")
#     if ENABLE_DISCONNECT_ALERTS: logging.info(f"Disconnect detection is ENABLED.")
#     if ENABLE_HARD_RESET: logging.warning(f"Hard reset is ENABLED. Command: '{HARD_RESET_COMMAND}'")
    
#     cap = None
#     # Freeze detection state
#     last_frame_signature, freeze_start_time, last_freeze_alert_time = None, None, None
#     # Disconnect detection state
#     connection_fail_start_time, last_disconnect_alert_time = None, None

#     while True:
#         try:
#             # --- Stream Connection Logic ---
#             if cap is None or not cap.isOpened():
#                 if ENABLE_DISCONNECT_ALERTS:
#                     if connection_fail_start_time is None:
#                         logging.warning("Failed to connect to stream. Starting disconnect timer...")
#                         connection_fail_start_time = time.time()
#                     if (time.time() - connection_fail_start_time) > DISCONNECT_THRESHOLD_SECONDS:
#                         time_to_alert = last_disconnect_alert_time is None or \
#                                        (time.time() - last_disconnect_alert_time) > REPEAT_DISCONNECT_SECONDS
#                         if time_to_alert:
#                             last_disconnect_alert_time = time.time()
#                             if ENABLE_HARD_RESET:
#                                 threading.Thread(target=execute_hard_reset, kwargs={"reason": "offline"}).start()
#                             else:
#                                 logging.critical("STREAM OFFLINE! Sending disconnect alert.")
#                                 if SEND_TELEGRAM_ALERT_ON_FREEZE:
#                                     threading.Thread(target=send_disconnect_alert_sync).start()
#                 logging.info("Attempting to connect to video stream...")
#                 cap = cv2.VideoCapture(STREAM_URL)
#                 if not cap.isOpened():
#                     time.sleep(10)
#                     continue
            
#             # --- Connection is now considered successful ---
#             if connection_fail_start_time is not None:
#                 logging.info("Stream connection has been RESTORED.")
#                 connection_fail_start_time, last_disconnect_alert_time = None, None

#             # --- Frame Reading & Freeze Detection ---
#             ret, frame = cap.read()
#             if not ret:
#                 logging.warning("Connected but failed to grab frame. Re-establishing...")
#                 cap.release(); cap = None
#                 time.sleep(5)
#                 continue

#             if ENABLE_FREEZE_DETECTOR:
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 current_signature = cv2.resize(gray, (128, 72), interpolation=cv2.INTER_AREA)
#                 if last_frame_signature is not None:
#                     diff_sum = int(np.sum(cv2.absdiff(last_frame_signature, current_signature)))
#                     log_message = f"Frame difference: {diff_sum} (Threshold: {FREEZE_SENSITIVITY_THRESHOLD})"
#                     if freeze_start_time is not None:
#                         frozen_for = int(time.time() - freeze_start_time)
#                         log_message += f" | Frozen for: {frozen_for}s"
#                     logging.info(log_message)

#                     if diff_sum < FREEZE_SENSITIVITY_THRESHOLD:
#                         if freeze_start_time is None: freeze_start_time = time.time()
                        
#                         # --- FINAL BACKSTOP LOGIC ---
#                         # Check if the connection has been stale for too long
#                         if (time.time() - freeze_start_time) > STALE_CONNECTION_THRESHOLD_SECONDS:
#                             logging.critical(f"STALE CONNECTION! Stream has been in a frozen state for over {STALE_CONNECTION_THRESHOLD_SECONDS}s. Forcing hard reset.")
#                             if ENABLE_HARD_RESET:
#                                 # Immediately execute the reset and break the loop to reconnect
#                                 execute_hard_reset(reason="stale")
#                                 freeze_start_time, last_freeze_alert_time = None, None
#                                 cap.release(); cap = None
#                                 continue # Force reconnection
#                             else:
#                                 # If hard reset is off, just keep sending normal alerts
#                                 pass # Fall through to normal alert logic

#                         # --- Normal Freeze Alert Logic ---
#                         if (time.time() - freeze_start_time) > FREEZE_DURATION_SECONDS:
#                             time_to_alert = last_freeze_alert_time is None or \
#                                            (ENABLE_REPEATED_ALERTS and (time.time() - last_freeze_alert_time) > REPEAT_ALERT_SECONDS)
#                             if time_to_alert:
#                                 last_freeze_alert_time = time.time()
#                                 if ENABLE_HARD_RESET:
#                                     threading.Thread(target=execute_hard_reset, kwargs={"reason": "frozen"}).start()
#                                 else:
#                                     logging.critical(f"STREAM FROZEN! Sending freeze alert.")
#                                     if SEND_TELEGRAM_ALERT_ON_FREEZE:
#                                         threading.Thread(target=send_freeze_alert_sync).start()
#                     else:
#                         if freeze_start_time is not None: logging.info("Stream freeze condition resolved.")
#                         freeze_start_time, last_freeze_alert_time = None, None
#                 last_frame_signature = current_signature
            
#             time.sleep(POLL_INTERVAL)

#         except KeyboardInterrupt:
#             logging.info("Watchdog stopped by user.")
#             break
#         except Exception as e:
#             logging.error(f"An unexpected error occurred: {e}", exc_info=True)
#             if cap: cap.release(); cap = None
#             time.sleep(10)

#     if cap: cap.release()
#     logging.info("Watchdog has shut down.")

# if __name__ == "__main__":
#     run_watchdog()

# # # watchdog.py
# # #
# # # v7 (Final): Standalone script where ALL time-based settings are in SECONDS.
# # #
# # # See `config.ini` for `[watchdog]` section to configure.
# # #
# # import cv2
# # import numpy as np
# # import time
# # import configparser
# # import logging
# # import sys
# # import os
# # import threading
# # import asyncio
# # import argparse

# # try:
# #     import telegram
# # except ImportError:
# #     print("Error: The 'python-telegram-bot' library is not installed.")
# #     print("Please install it using: pip install 'python-telegram-bot==13.15'")
# #     sys.exit(1)

# # # --- Basic Logging Setup ---
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='%(asctime)s - %(levelname)s - [Watchdog] - %(message)s',
# #     stream=sys.stdout,
# # )

# # # --- Self-Contained Telegram Functions ---
# # def _send_telegram_alert(message):
# #     """Generic alert sender to avoid code duplication."""
# #     async def send_async():
# #         bot_token = os.getenv('DVR_YOLOV8_TELEGRAM_BOT_TOKEN')
# #         user_ids_str = os.getenv('DVR_YOLOV8_ALLOWED_TELEGRAM_USERS')
# #         if not bot_token or not user_ids_str:
# #             logging.error("Cannot send Telegram alert. Environment variables for the bot are not set.")
# #             return
# #         bot = telegram.Bot(token=bot_token)
# #         user_ids = [int(uid.strip()) for uid in user_ids_str.split(',')]
# #         for user_id in user_ids:
# #             try:
# #                 await bot.send_message(chat_id=user_id, text=message, parse_mode="HTML")
# #             except Exception as e:
# #                 logging.error(f"Failed to send message to user {user_id}: {e}")
# #     try:
# #         asyncio.run(send_async())
# #         logging.info("Successfully sent alert to all allowed users.")
# #     except Exception as e:
# #         logging.error(f"An error occurred in the Telegram alert thread: {e}")

# # def send_freeze_alert_sync():
# #     _send_telegram_alert(
# #         "🚨❄️ <b>STREAM FROZEN</b> ❄️🚨\n\n"
# #         "The watchdog is connected, but the video stream has been static for a significant duration.\n"
# #         "Please check the source (e.g., OBS)."
# #     )

# # def send_disconnect_alert_sync():
# #     _send_telegram_alert(
# #         "🚨‼️ <b>STREAM OFFLINE</b> ‼️🚨\n\n"
# #         "The watchdog cannot connect to the stream source.\n"
# #         "Please check your streaming software or network connection."
# #     )

# # # --- Main Application ---
# # try:
# #     config = configparser.ConfigParser(interpolation=None)
# #     config.read('config.ini')
# #     STREAM_URL = config.get('stream', 'stream_url', fallback='rtmp://127.0.0.1:1935/live/stream')
    
# #     # --- THIS IS THE FIX: All time settings are now read directly in seconds ---
# #     POLL_INTERVAL = config.getint('watchdog', 'poll_interval_seconds', fallback=1)
# #     SEND_TELEGRAM_ALERT_ON_FREEZE = config.getboolean('watchdog', 'send_telegram_alert_on_freeze', fallback=True)
# #     # Freeze Detection
# #     ENABLE_FREEZE_DETECTOR = config.getboolean('watchdog', 'enable_freeze_detector', fallback=True)
# #     FREEZE_DURATION_SECONDS = config.getint('watchdog', 'freeze_duration_seconds', fallback=10)
# #     FREEZE_SENSITIVITY_THRESHOLD = config.getint('watchdog', 'freeze_sensitivity_threshold', fallback=250)
# #     ENABLE_REPEATED_ALERTS = config.getboolean('watchdog', 'enable_repeated_alerts', fallback=True)
# #     REPEAT_ALERT_SECONDS = config.getint('watchdog', 'repeat_alert_interval_seconds', fallback=60)
# #     # Disconnect Detection
# #     ENABLE_DISCONNECT_ALERTS = config.getboolean('watchdog', 'enable_disconnect_alerts', fallback=True)
# #     DISCONNECT_THRESHOLD_SECONDS = config.getint('watchdog', 'disconnect_alert_threshold_seconds', fallback=60)
# #     REPEAT_DISCONNECT_SECONDS = config.getint('watchdog', 'repeat_disconnect_alert_interval_seconds', fallback=60)

# # except Exception as e:
# #     logging.error(f"FATAL: Could not read config.ini. Error: {e}")
# #     sys.exit(1)

# # def run_watchdog():
# #     logging.info("Starting standalone stream watchdog...")
# #     if ENABLE_FREEZE_DETECTOR: logging.info(f"Freeze detection is ENABLED.")
# #     if ENABLE_DISCONNECT_ALERTS: logging.info(f"Disconnect detection is ENABLED.")
    
# #     cap, stream_is_confirmed_up = None, False
# #     last_frame_signature, freeze_start_time, last_freeze_alert_time = None, None, None
# #     connection_fail_start_time, last_disconnect_alert_time = None, None

# #     while True:
# #         try:
# #             # --- Stream Connection Logic ---
# #             if cap is None or not cap.isOpened():
# #                 if stream_is_confirmed_up and ENABLE_DISCONNECT_ALERTS:
# #                     if connection_fail_start_time is None:
# #                         logging.warning("Lost connection to active stream. Starting disconnect timer...")
# #                         connection_fail_start_time = time.time()
# #                     if (time.time() - connection_fail_start_time) > DISCONNECT_THRESHOLD_SECONDS:
# #                         time_to_alert = last_disconnect_alert_time is None or \
# #                                        (time.time() - last_disconnect_alert_time) > REPEAT_DISCONNECT_SECONDS
# #                         if time_to_alert:
# #                             logging.critical("STREAM OFFLINE! Sending disconnect alert.")
# #                             if SEND_TELEGRAM_ALERT_ON_FREEZE:
# #                                 threading.Thread(target=send_disconnect_alert_sync).start()
# #                             last_disconnect_alert_time = time.time()
# #                 logging.info("Attempting to connect to video stream...")
# #                 cap = cv2.VideoCapture(STREAM_URL)
# #                 if not cap.isOpened():
# #                     time.sleep(10)
# #                     continue
            
# #             # --- Connection is now considered successful ---
# #             if not stream_is_confirmed_up:
# #                 logging.info("Stream connection is active.")
# #                 stream_is_confirmed_up = True
# #             if connection_fail_start_time is not None:
# #                 logging.info("Stream connection has been RESTORED.")
# #                 connection_fail_start_time, last_disconnect_alert_time = None, None

# #             # --- Frame Reading & Freeze Detection ---
# #             ret, frame = cap.read()
# #             if not ret:
# #                 logging.warning("Connected but failed to grab frame. Re-establishing...")
# #                 cap.release(); cap = None
# #                 time.sleep(5)
# #                 continue

# #             if ENABLE_FREEZE_DETECTOR:
# #                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #                 current_signature = cv2.resize(gray, (128, 72), interpolation=cv2.INTER_AREA)
# #                 if last_frame_signature is not None:
# #                     diff_sum = int(np.sum(cv2.absdiff(last_frame_signature, current_signature)))
                    
# #                     log_message = f"Frame difference: {diff_sum} (Threshold: {FREEZE_SENSITIVITY_THRESHOLD})"
# #                     if freeze_start_time is not None:
# #                         frozen_for = int(time.time() - freeze_start_time)
# #                         log_message += f" | Frozen for: {frozen_for}s"
# #                     logging.info(log_message)

# #                     if diff_sum < FREEZE_SENSITIVITY_THRESHOLD:
# #                         if freeze_start_time is None: freeze_start_time = time.time()
# #                         if (time.time() - freeze_start_time) > FREEZE_DURATION_SECONDS:
# #                             time_to_alert = last_freeze_alert_time is None or \
# #                                            (ENABLE_REPEATED_ALERTS and (time.time() - last_freeze_alert_time) > REPEAT_ALERT_SECONDS)
# #                             if time_to_alert:
# #                                 logging.critical(f"STREAM FROZEN! Sending freeze alert.")
# #                                 if SEND_TELEGRAM_ALERT_ON_FREEZE:
# #                                     threading.Thread(target=send_freeze_alert_sync).start()
# #                                 last_freeze_alert_time = time.time()
# #                     else:
# #                         if freeze_start_time is not None: logging.info("Stream freeze condition resolved.")
# #                         freeze_start_time, last_freeze_alert_time = None, None
# #                 last_frame_signature = current_signature
            
# #             time.sleep(POLL_INTERVAL)

# #         except KeyboardInterrupt:
# #             logging.info("Watchdog stopped by user.")
# #             break
# #         except Exception as e:
# #             logging.error(f"An unexpected error occurred: {e}", exc_info=True)
# #             if cap: cap.release(); cap = None
# #             time.sleep(10)

# #     if cap: cap.release()
# #     logging.info("Watchdog has shut down.")

# # if __name__ == "__main__":
# #     run_watchdog()

# # # # watchdog.py
# # # #
# # # # v5 (Final): Standalone script with REPEATED alerts for both
# # # #     FROZEN and OFFLINE streams, with CONSTANT console feedback restored.
# # # #
# # # # See `config.ini` for `[watchdog]` section to configure.
# # # #
# # # # To run normally:  python3 watchdog.py
# # # #
# # # import cv2
# # # import numpy as np
# # # import time
# # # import configparser
# # # import logging
# # # import sys
# # # import os
# # # import threading
# # # import asyncio
# # # import argparse

# # # try:
# # #     import telegram
# # # except ImportError:
# # #     print("Error: The 'python-telegram-bot' library is not installed.")
# # #     print("Please install it using: pip install 'python-telegram-bot==13.15'")
# # #     sys.exit(1)

# # # # --- Basic Logging Setup ---
# # # logging.basicConfig(
# # #     level=logging.INFO,
# # #     format='%(asctime)s - %(levelname)s - [Watchdog] - %(message)s',
# # #     stream=sys.stdout,
# # # )

# # # # --- Self-Contained Telegram Functions ---
# # # def _send_telegram_alert(message):
# # #     """Generic alert sender to avoid code duplication."""
# # #     async def send_async():
# # #         bot_token = os.getenv('DVR_YOLOV8_TELEGRAM_BOT_TOKEN')
# # #         user_ids_str = os.getenv('DVR_YOLOV8_ALLOWED_TELEGRAM_USERS')
# # #         if not bot_token or not user_ids_str:
# # #             logging.error("Cannot send Telegram alert. Environment variables for the bot are not set.")
# # #             return
# # #         bot = telegram.Bot(token=bot_token)
# # #         user_ids = [int(uid.strip()) for uid in user_ids_str.split(',')]
# # #         for user_id in user_ids:
# # #             try:
# # #                 await bot.send_message(chat_id=user_id, text=message, parse_mode="HTML")
# # #             except Exception as e:
# # #                 logging.error(f"Failed to send message to user {user_id}: {e}")
# # #     try:
# # #         asyncio.run(send_async())
# # #         logging.info("Successfully sent alert to all allowed users.")
# # #     except Exception as e:
# # #         logging.error(f"An error occurred in the Telegram alert thread: {e}")

# # # def send_freeze_alert_sync():
# # #     """Sends the 'Stream Frozen' message."""
# # #     _send_telegram_alert(
# # #         "🚨❄️ <b>STREAM FROZEN</b> ❄️🚨\n\n"
# # #         "The watchdog is connected, but the video stream has been static for a significant duration.\n"
# # #         "Please check the source (e.g., OBS)."
# # #     )

# # # def send_disconnect_alert_sync():
# # #     """Sends the 'Stream Offline' message."""
# # #     _send_telegram_alert(
# # #         "🚨‼️ <b>STREAM OFFLINE</b> ‼️🚨\n\n"
# # #         "The watchdog cannot connect to the stream source.\n"
# # #         "Please check your streaming software or network connection."
# # #     )

# # # # --- Main Application ---
# # # try:
# # #     config = configparser.ConfigParser(interpolation=None)
# # #     config.read('config.ini')
# # #     STREAM_URL = config.get('stream', 'stream_url', fallback='rtmp://127.0.0.1:1935/live/stream')
# # #     # Freeze Detection
# # #     ENABLE_FREEZE_DETECTOR = config.getboolean('watchdog', 'enable_freeze_detector', fallback=True)
# # #     FREEZE_DURATION = config.getint('watchdog', 'freeze_duration', fallback=10)
# # #     FREEZE_SENSITIVITY_THRESHOLD = config.getint('watchdog', 'freeze_sensitivity_threshold', fallback=500)
# # #     SEND_TELEGRAM_ALERT_ON_FREEZE = config.getboolean('watchdog', 'send_telegram_alert_on_freeze', fallback=True)
# # #     POLL_INTERVAL = config.getint('watchdog', 'poll_interval_seconds', fallback=1)
# # #     ENABLE_REPEATED_ALERTS = config.getboolean('watchdog', 'enable_repeated_alerts', fallback=True)
# # #     REPEAT_ALERT_INTERVAL = config.getint('watchdog', 'repeat_alert_interval_minutes', fallback=5)
# # #     REPEAT_ALERT_SECONDS = REPEAT_ALERT_INTERVAL * 60
# # #     # Disconnect Detection
# # #     ENABLE_DISCONNECT_ALERTS = config.getboolean('watchdog', 'enable_disconnect_alerts', fallback=True)
# # #     DISCONNECT_THRESHOLD = config.getint('watchdog', 'disconnect_alert_threshold_minutes', fallback=5)
# # #     DISCONNECT_THRESHOLD_SECONDS = DISCONNECT_THRESHOLD * 60
# # #     REPEAT_DISCONNECT_INTERVAL = config.getint('watchdog', 'repeat_disconnect_alert_interval_minutes', fallback=15)
# # #     REPEAT_DISCONNECT_SECONDS = REPEAT_DISCONNECT_INTERVAL * 60
# # # except Exception as e:
# # #     logging.error(f"FATAL: Could not read config.ini. Error: {e}")
# # #     sys.exit(1)

# # # def run_watchdog():
# # #     logging.info("Starting standalone stream watchdog...")
# # #     if ENABLE_FREEZE_DETECTOR: logging.info(f"Freeze detection is ENABLED.")
# # #     if ENABLE_DISCONNECT_ALERTS: logging.info(f"Disconnect detection is ENABLED.")
    
# # #     cap, stream_is_confirmed_up = None, False
# # #     last_frame_signature, freeze_start_time, last_freeze_alert_time = None, None, None
# # #     connection_fail_start_time, last_disconnect_alert_time = None, None

# # #     while True:
# # #         try:
# # #             # --- Stream Connection Logic ---
# # #             if cap is None or not cap.isOpened():
# # #                 if stream_is_confirmed_up and ENABLE_DISCONNECT_ALERTS:
# # #                     if connection_fail_start_time is None:
# # #                         logging.warning("Lost connection to active stream. Starting disconnect timer...")
# # #                         connection_fail_start_time = time.time()
# # #                     if (time.time() - connection_fail_start_time) > DISCONNECT_THRESHOLD_SECONDS:
# # #                         time_to_alert = last_disconnect_alert_time is None or \
# # #                                        (time.time() - last_disconnect_alert_time) > REPEAT_DISCONNECT_SECONDS
# # #                         if time_to_alert:
# # #                             logging.critical("STREAM OFFLINE! Sending disconnect alert.")
# # #                             if SEND_TELEGRAM_ALERT_ON_FREEZE:
# # #                                 threading.Thread(target=send_disconnect_alert_sync).start()
# # #                             last_disconnect_alert_time = time.time()
# # #                 logging.info("Attempting to connect to video stream...")
# # #                 cap = cv2.VideoCapture(STREAM_URL)
# # #                 if not cap.isOpened():
# # #                     time.sleep(10)
# # #                     continue
            
# # #             # --- Connection is now considered successful ---
# # #             if not stream_is_confirmed_up:
# # #                 logging.info("Stream connection is active.")
# # #                 stream_is_confirmed_up = True
# # #             if connection_fail_start_time is not None:
# # #                 logging.info("Stream connection has been RESTORED.")
# # #                 connection_fail_start_time, last_disconnect_alert_time = None, None

# # #             # --- Frame Reading & Freeze Detection ---
# # #             ret, frame = cap.read()
# # #             if not ret:
# # #                 logging.warning("Connected but failed to grab frame. Re-establishing...")
# # #                 cap.release(); cap = None
# # #                 time.sleep(5)
# # #                 continue

# # #             if ENABLE_FREEZE_DETECTOR:
# # #                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # #                 current_signature = cv2.resize(gray, (128, 72), interpolation=cv2.INTER_AREA)
# # #                 if last_frame_signature is not None:
# # #                     diff_sum = int(np.sum(cv2.absdiff(last_frame_signature, current_signature)))
                    
# # #                     # --- THIS IS THE RESTORED LOGGING, IT WILL RUN EVERY TIME ---
# # #                     log_message = f"Frame difference: {diff_sum} (Threshold: {FREEZE_SENSITIVITY_THRESHOLD})"
# # #                     if freeze_start_time is not None:
# # #                         frozen_for = int(time.time() - freeze_start_time)
# # #                         log_message += f" | Frozen for: {frozen_for}s"
# # #                     logging.info(log_message)
# # #                     # --- END RESTORED LOGGING ---

# # #                     if diff_sum < FREEZE_SENSITIVITY_THRESHOLD:
# # #                         if freeze_start_time is None: freeze_start_time = time.time()
# # #                         if (time.time() - freeze_start_time) > FREEZE_DURATION:
# # #                             time_to_alert = last_freeze_alert_time is None or \
# # #                                            (ENABLE_REPEATED_ALERTS and (time.time() - last_freeze_alert_time) > REPEAT_ALERT_SECONDS)
# # #                             if time_to_alert:
# # #                                 logging.critical(f"STREAM FROZEN! Sending freeze alert.")
# # #                                 if SEND_TELEGRAM_ALERT_ON_FREEZE:
# # #                                     threading.Thread(target=send_freeze_alert_sync).start()
# # #                                 last_freeze_alert_time = time.time()
# # #                     else:
# # #                         if freeze_start_time is not None: logging.info("Stream freeze condition resolved.")
# # #                         freeze_start_time, last_freeze_alert_time = None, None
# # #                 last_frame_signature = current_signature
            
# # #             time.sleep(POLL_INTERVAL)

# # #         except KeyboardInterrupt:
# # #             logging.info("Watchdog stopped by user.")
# # #             break
# # #         except Exception as e:
# # #             logging.error(f"An unexpected error occurred: {e}", exc_info=True)
# # #             if cap: cap.release(); cap = None
# # #             time.sleep(10)

# # #     if cap: cap.release()
# # #     logging.info("Watchdog has shut down.")

# # # if __name__ == "__main__":
# # #     run_watchdog()

# # # ## // old
# # # # # watchdog.py
# # # # #
# # # # # v3: A completely standalone script with REPEATED ALERTS.
# # # # #     - See `config.ini` for `[watchdog]` section to configure.
# # # # #
# # # # # To run normally:  python3 watchdog.py
# # # # # To debug/tune:     python3 watchdog.py --debug
# # # # #

# # # # import cv2
# # # # import numpy as np
# # # # import time
# # # # import configparser
# # # # import logging
# # # # import sys
# # # # import os
# # # # import threading
# # # # import asyncio
# # # # import argparse

# # # # try:
# # # #     import telegram
# # # # except ImportError:
# # # #     print("Error: The 'python-telegram-bot' library is not installed.")
# # # #     print("Please install it using: pip install 'python-telegram-bot==13.15'")
# # # #     sys.exit(1)

# # # # # --- Basic Logging Setup ---
# # # # logging.basicConfig(
# # # #     level=logging.INFO,
# # # #     format='%(asctime)s - %(levelname)s - [Watchdog] - %(message)s',
# # # #     stream=sys.stdout,
# # # # )

# # # # # --- Self-Contained Telegram Function ---
# # # # def send_telegram_alert_sync():
# # # #     try:
# # # #         asyncio.run(send_telegram_alert_async())
# # # #         logging.info("Successfully sent stream freeze alert to all allowed users.")
# # # #     except Exception as e:
# # # #         logging.error(f"An error occurred in the Telegram alert thread: {e}")

# # # # async def send_telegram_alert_async():
# # # #     bot_token = os.getenv('DVR_YOLOV8_TELEGRAM_BOT_TOKEN')
# # # #     user_ids_str = os.getenv('DVR_YOLOV8_ALLOWED_TELEGRAM_USERS')
# # # #     if not bot_token or not user_ids_str:
# # # #         logging.error("Cannot send Telegram alert. Environment variables for the bot are not set.")
# # # #         return

# # # #     bot = telegram.Bot(token=bot_token)
# # # #     user_ids = [int(uid.strip()) for uid in user_ids_str.split(',')]
    
# # # #     freeze_message = (
# # # #         "🚨❄️ <b>STREAM FROZEN</b> ❄️🚨\n\n"
# # # #         "The watchdog has detected that the video stream has been frozen for a significant duration.\n"
# # # #         "Please check the source (e.g., OBS)."
# # # #     )

# # # #     for user_id in user_ids:
# # # #         try:
# # # #             await bot.send_message(chat_id=user_id, text=freeze_message, parse_mode="HTML")
# # # #         except Exception as e:
# # # #             logging.error(f"Failed to send message to user {user_id}: {e}")

# # # # # --- Main Application ---
# # # # try:
# # # #     config = configparser.ConfigParser(interpolation=None)
# # # #     config.read('config.ini')

# # # #     STREAM_URL = config.get('stream', 'stream_url', fallback='rtmp://127.0.0.1:1935/live/stream')
# # # #     ENABLE_FREEZE_DETECTOR = config.getboolean('watchdog', 'enable_freeze_detector', fallback=False)
# # # #     FREEZE_DURATION = config.getint('watchdog', 'freeze_duration', fallback=10)
# # # #     FREEZE_SENSITIVITY_THRESHOLD = config.getint('watchdog', 'freeze_sensitivity_threshold', fallback=500)
# # # #     SEND_TELEGRAM_ALERT_ON_FREEZE = config.getboolean('watchdog', 'send_telegram_alert_on_freeze', fallback=True)
# # # #     POLL_INTERVAL = config.getint('watchdog', 'poll_interval_seconds', fallback=1)
    
# # # #     # --- Load New Repeat Alert Settings ---
# # # #     ENABLE_REPEATED_ALERTS = config.getboolean('watchdog', 'enable_repeated_alerts', fallback=True)
# # # #     REPEAT_ALERT_INTERVAL = config.getint('watchdog', 'repeat_alert_interval_minutes', fallback=1)
# # # #     # Convert minutes to seconds for internal use
# # # #     REPEAT_ALERT_SECONDS = REPEAT_ALERT_INTERVAL * 60

# # # # except Exception as e:
# # # #     logging.error(f"FATAL: Could not read config.ini. Error: {e}")
# # # #     sys.exit(1)

# # # # def run_watchdog(debug_mode=False):
# # # #     if not ENABLE_FREEZE_DETECTOR:
# # # #         logging.info("Freeze detector is disabled in config.ini. Exiting.")
# # # #         return

# # # #     logging.info("Starting standalone stream freeze watchdog...")
# # # #     if ENABLE_REPEATED_ALERTS:
# # # #         logging.info(f"Repeat alerts are ENABLED and will be sent every {REPEAT_ALERT_INTERVAL} minute(s) if stream remains frozen.")
# # # #     else:
# # # #         logging.info("Repeat alerts are DISABLED.")


# # # #     last_frame_signature = None
# # # #     freeze_start_time = None
# # # #     # --- This now tracks the time of the last alert ---
# # # #     last_alert_time = None
# # # #     cap = None

# # # #     while True:
# # # #         try:
# # # #             if cap is None or not cap.isOpened():
# # # #                 logging.info("Attempting to connect to video stream...")
# # # #                 cap = cv2.VideoCapture(STREAM_URL)
# # # #                 if not cap.isOpened():
# # # #                     logging.warning("Failed to connect. Retrying in 10 seconds...")
# # # #                     time.sleep(10)
# # # #                     continue
# # # #                 logging.info("Successfully connected.")

# # # #             ret, frame = cap.read()
# # # #             if not ret:
# # # #                 logging.warning("Failed to grab a frame. Reconnecting...")
# # # #                 cap.release()
# # # #                 cap = None
# # # #                 time.sleep(5)
# # # #                 continue

# # # #             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # # #             current_signature = cv2.resize(gray_frame, (128, 72), interpolation=cv2.INTER_AREA)

# # # #             if last_frame_signature is not None:
# # # #                 diff_image = cv2.absdiff(last_frame_signature, current_signature)
# # # #                 diff_sum = int(np.sum(diff_image))

# # # #                 log_message = f"Frame difference: {diff_sum} (Threshold: {FREEZE_SENSITIVITY_THRESHOLD})"
# # # #                 if freeze_start_time is not None:
# # # #                     frozen_for = int(time.time() - freeze_start_time)
# # # #                     log_message += f" | Frozen for: {frozen_for}s"
# # # #                 logging.info(log_message)

# # # #                 if debug_mode:
# # # #                     cv2.imwrite("watchdog_debug/current.png", current_signature)
# # # #                     cv2.imwrite("watchdog_debug/last.png", last_frame_signature)
# # # #                     cv2.imwrite("watchdog_debug/diff.png", diff_image)

# # # #                 # --- Main Freeze Detection Logic ---
# # # #                 if diff_sum < FREEZE_SENSITIVITY_THRESHOLD:
# # # #                     if freeze_start_time is None:
# # # #                         freeze_start_time = time.time()
                    
# # # #                     # --- UPGRADED Alerting Logic ---
# # # #                     # Check if the initial freeze duration has passed
# # # #                     if (time.time() - freeze_start_time) > FREEZE_DURATION:
                        
# # # #                         # Determine if it's time to send an alert
# # # #                         time_to_alert = False
# # # #                         if last_alert_time is None:
# # # #                             # If we've never sent an alert for this event, send one now.
# # # #                             time_to_alert = True
# # # #                         elif ENABLE_REPEATED_ALERTS and (time.time() - last_alert_time) > REPEAT_ALERT_SECONDS:
# # # #                             # If repeat alerts are on and enough time has passed since the last one.
# # # #                             time_to_alert = True

# # # #                         if time_to_alert:
# # # #                             logging.critical(f"STREAM FROZEN! Sending alert.")
# # # #                             if SEND_TELEGRAM_ALERT_ON_FREEZE:
# # # #                                 threading.Thread(target=send_telegram_alert_sync).start()
# # # #                             # Record the time of this alert
# # # #                             last_alert_time = time.time()
# # # #                 else:
# # # #                     # Stream has recovered
# # # #                     if freeze_start_time is not None:
# # # #                         logging.info("Stream is active again.")
# # # #                     freeze_start_time = None
# # # #                     last_alert_time = None
            
# # # #             last_frame_signature = current_signature
# # # #             time.sleep(POLL_INTERVAL)

# # # #         except KeyboardInterrupt:
# # # #             logging.info("Watchdog stopped by user.")
# # # #             break
# # # #         except Exception as e:
# # # #             logging.error(f"An unexpected error occurred: {e}", exc_info=True)
# # # #             if cap: cap.release()
# # # #             cap = None
# # # #             time.sleep(10)

# # # #     if cap: cap.release()
# # # #     logging.info("Watchdog has shut down.")

# # # # if __name__ == "__main__":
# # # #     parser = argparse.ArgumentParser(description="Standalone watchdog for detecting frozen RTMP streams.")
# # # #     parser.add_argument("--debug", action="store_true", help="Run in debug mode to save frame comparisons as images.")
# # # #     args = parser.parse_args()
    
# # # #     run_watchdog(debug_mode=args.debug)