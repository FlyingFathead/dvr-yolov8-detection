# telegram_alerts.py

# => Telegram alerts module for DVR-YOLOv8-Detection <=
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/dvr-yolov8-detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Version number
import version  # Import the version module
version_number = version.version_number

import configparser
import os
import threading
import time
import logging
import telegram
import asyncio
from datetime import datetime

# Configure logging
logger = logging.getLogger('telegram_alerts')
logger.setLevel(logging.INFO)

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

TELEGRAM_MESSAGE_LIMIT = 4096  # Telegram's message size limit

# Thresholds and intervals from config
AGGREGATION_INTERVAL = config.getfloat('telegram', 'aggregation_interval', fallback=1.0)
DETECTION_COOLDOWN = config.getfloat('telegram', 'detection_cooldown', fallback=30.0)
ENABLE_IMMEDIATE_ALERTS = config.getboolean('telegram', 'enable_immediate_alerts', fallback=True)
CONFIDENCE_WARNING_THRESHOLD = config.getfloat('telegram', 'confidence_warning_threshold', fallback=0.68)
DETECTION_COUNT_WARNING_THRESHOLD = config.getint('telegram', 'detection_count_warning_threshold', fallback=5)

# Read environment variables
BOT_TOKEN = os.getenv('DVR_YOLOV8_TELEGRAM_BOT_TOKEN')
ALLOWED_USERS = os.getenv('DVR_YOLOV8_ALLOWED_TELEGRAM_USERS')

# Configure the maximum number of concurrent connections allowed
MAX_CONCURRENT_SENDS = 1  # Adjust based on your needs and Telegram's rate limits

# Initialize bot and allowed users
bot = None
allowed_users = []
loop = None
send_semaphore = None  # Will be initialized in the event loop

if BOT_TOKEN and ALLOWED_USERS:
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        allowed_users = [int(uid.strip()) for uid in ALLOWED_USERS.split(',')]
        logger.info(f"Telegram bot initialized with allowed users: {allowed_users}")
    except Exception as e:
        logger.error(f"Error initializing Telegram bot: {e}")
else:
    logger.warning("Telegram bot token or allowed users not set. Telegram alerts will be disabled.")

# Start an asyncio event loop in a new thread
def start_event_loop():
    global loop, send_semaphore
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # Initialize the asyncio.Semaphore
    send_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SENDS)
    loop.run_forever()

event_loop_thread = threading.Thread(target=start_event_loop, daemon=True)
event_loop_thread.start()

def split_message(message):
    """Split a long message into chunks to respect Telegram's character limit."""
    return [message[i:i + TELEGRAM_MESSAGE_LIMIT] for i in range(0, len(message), TELEGRAM_MESSAGE_LIMIT)]

async def send_message(user_id, message):
    """Send a single message or split messages to a user."""
    message_chunks = split_message(message)
    for chunk in message_chunks:
        try:
            # Use async with for asyncio.Semaphore
            async with send_semaphore:
                await bot.send_message(chat_id=user_id, text=chunk, parse_mode="HTML")
                logger.info(f"Sent message to user {user_id}: {chunk[:50]}...")
        except Exception as e:
            logger.error(f"Error sending message to {user_id}: {e}")

def send_alert(message):
    """Send an alert message to all allowed users."""
    if bot and allowed_users:
        for user_id in allowed_users:
            asyncio.run_coroutine_threadsafe(
                send_message(user_id, message), loop
            )
    else:
        logger.warning("Bot not initialized or no allowed users. Cannot send message.")

def send_summary_alert(detections):
    """Generate and send a summary message for aggregated detections."""
    try:
        first_seen = min(d['timestamp'] for d in detections if isinstance(d, dict))
        last_seen = max(d['timestamp'] for d in detections if isinstance(d, dict))
        min_confidence = min(d['confidence'] for d in detections if isinstance(d, dict))
        max_confidence = max(d['confidence'] for d in detections if isinstance(d, dict))
        count = len(detections)
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error generating summary: {e}")
        return  # Return if there are issues

    warning = '⚠️' if max_confidence >= CONFIDENCE_WARNING_THRESHOLD else ''
    bold_count = f"<strong>{count}</strong>" if count >= DETECTION_COUNT_WARNING_THRESHOLD else f"{count}"

    summary = (
        f"👀 <b>Detection Summary</b> {warning}\n"
        f"<b>First Seen:</b> {first_seen}\n"
        f"<b>Last Seen:</b> {last_seen}\n"
        f"<b>Number of Detections:</b> {bold_count}\n"
        f"<b>Confidence Range:</b> {min_confidence:.2f} - {max_confidence:.2f}"
    )
    logger.info("Sending summary alert...")
    send_alert(summary)

def queue_alert(detection):
    """Queue a detection alert and manage cooldown for summary."""
    global last_detection_time
    logger.info(f"Received detection alert: {detection} (Type: {type(detection)})")

    if not isinstance(detection, dict):
        logger.error("Detection is not a dictionary! Check the caller.")
        return  # Exit early to avoid errors

    with lock:
        pending_detections.append(detection)
        last_detection_time = time.time()
        detection_received_event.set()  # Signal that a detection has been received

    if ENABLE_IMMEDIATE_ALERTS:
        # Aggregate immediate alerts over AGGREGATION_INTERVAL
        with immediate_lock:
            immediate_alerts.append(detection)

def immediate_alert_sender():
    """Send immediate alerts, optionally aggregated over a short interval."""
    while True:
        time.sleep(AGGREGATION_INTERVAL)
        with immediate_lock:
            if immediate_alerts:
                alerts_to_send = immediate_alerts.copy()
                immediate_alerts.clear()

                detections_info = "\n".join([
                    # f"🚨 Detection {d.get('frame_count', 'N/A')} at {d.get('timestamp', 'N/A')}\n"
                    f"🚨 Detection at {d.get('timestamp', 'N/A')}\n"
                    f"Coordinates: {d.get('coordinates', 'N/A')}\n"
                    f"Confidence: {d.get('confidence', 0.0):.2f}"
                    for d in alerts_to_send
                ])

                message = f"{detections_info}"
                send_alert(message)

def summary_alert_sender():
    """Send a summary alert after the cooldown period with no detections."""
    global last_detection_time
    while True:
        detection_received_event.wait()  # Wait until a detection is received
        # Wait for cooldown period
        time.sleep(DETECTION_COOLDOWN)
        current_time = time.time()
        with lock:
            time_since_last_detection = current_time - last_detection_time
            if time_since_last_detection >= DETECTION_COOLDOWN and pending_detections:
                # Send summary alert
                send_summary_alert(pending_detections)
                pending_detections.clear()
                detection_received_event.clear()  # Reset the event

# Initialize message aggregation
last_detection_time = 0
pending_detections = []
immediate_alerts = []
lock = threading.Lock()
immediate_lock = threading.Lock()
detection_received_event = threading.Event()

# Start the immediate alert sender thread
if ENABLE_IMMEDIATE_ALERTS:
    immediate_alert_thread = threading.Thread(target=immediate_alert_sender, daemon=True)
    immediate_alert_thread.start()

# Start the summary alert sender thread
summary_alert_thread = threading.Thread(target=summary_alert_sender, daemon=True)
summary_alert_thread.start()

def send_startup_message():
    """Send a startup notification message to all allowed users."""
    local_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    utc_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    startup_message = (
        f"🚀 YOLOv8 Detection Framework started.\n"
        f"Version number: v{version_number}\n"
        f"Local Time: {local_time}\nUTC Time: {utc_time}"
    )
    logger.info("Sending startup message...")
    send_alert(startup_message)

# Send the startup message on launch
if bot and allowed_users:
    send_startup_message()
else:
    logger.warning("Startup message could not be sent: Bot not initialized or no allowed users.")
