# detection_audio_poller.py
# Polls the local web server's API for detections and announces them via TTS

import requests
import time
import pyttsx3
import logging

# Configure the web server endpoint and polling interval
API_URL = 'http://127.0.0.1:5000/api/detections'
POLLING_INTERVAL = 1  # in seconds
# Startup message
STARTUP_MESSAGE = "DVR framework audio poller started."

# Configure logging to include timestamps and use local time
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.Formatter.converter = time.localtime

logger = logging.getLogger(__name__)

def get_latest_detection():
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            detections = response.json()
            if detections:
                # Assuming the latest detection is the first in the list
                return detections[0]
        return None
    except Exception as e:
        logger.error(f"Error fetching detections: {e}")
        return None

def main():
    engine = pyttsx3.init()
    last_detection_timestamp = None  # Using timestamp

    # Log and announce the startup message if it's set
    if STARTUP_MESSAGE:
        logger.info(STARTUP_MESSAGE)
        engine.say(STARTUP_MESSAGE)
        engine.runAndWait()

    # Initialize last_detection_timestamp to the current latest detection
    latest_detection = get_latest_detection()
    if latest_detection:
        logger.info(f"Latest detection data: {latest_detection}")
        # Use the correct key based on your data structure
        last_detection_timestamp = latest_detection.get('latest_timestamp')
        logger.info(f"Initialized last_detection_timestamp to {last_detection_timestamp}")
    else:
        logger.info("No detections available at startup.")

    # check if we're on first pass to change the welcome message
    first_pass = True

    while True:
        latest_detection = get_latest_detection()
        if latest_detection:
            detection_timestamp = latest_detection.get('latest_timestamp')
            if detection_timestamp != last_detection_timestamp:
                # Skip TTS on first pass, just update our reference
                if not first_pass:
                    engine.say("Human detected!")
                    engine.runAndWait()
                last_detection_timestamp = detection_timestamp
        first_pass = False
        time.sleep(POLLING_INTERVAL)

    # while True:
    #     latest_detection = get_latest_detection()
    #     if latest_detection:
    #         detection_timestamp = latest_detection.get('latest_timestamp')
    #         if detection_timestamp != last_detection_timestamp:
    #             # New detection found
    #             logger.info(f"New detection detected at {detection_timestamp}!")
    #             engine.say("Human detected!")
    #             engine.runAndWait()
    #             last_detection_timestamp = detection_timestamp
    #     else:
    #         logger.debug("No detections received.")
    #     time.sleep(POLLING_INTERVAL)

if __name__ == "__main__":
    main()
