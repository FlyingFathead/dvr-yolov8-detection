# audio_alert_test.py
# (for testing audio alerts)

import pyttsx3
import logging

def test_tts():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TTS_Test")

    try:
        # Initialize TTS engine
        engine = pyttsx3.init()
        logger.info("Initialized pyttsx3 engine.")

        # List available voices
        voices = engine.getProperty('voices')
        logger.info("Available voices:")
        for i, voice in enumerate(voices):
            logger.info(f"{i}: {voice.name} - {voice.id}")

        # Optional: Set a specific voice (uncomment if needed)
        # engine.setProperty('voice', voices[0].id)

        # Set speech rate
        rate = engine.getProperty('rate')
        logger.info(f"Current speech rate: {rate}")
        engine.setProperty('rate', rate - 25)  # Decrease speech rate

        # Set volume (0.0 to 1.0)
        volume = engine.getProperty('volume')
        logger.info(f"Current volume: {volume}")
        engine.setProperty('volume', 1.0)  # Set to max volume

        # Speak the test message
        test_message = "Human detected!"
        logger.info(f"Attempting to speak: '{test_message}'")
        engine.say(test_message)
        engine.runAndWait()
        logger.info("TTS announcement completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during TTS testing: {e}")

if __name__ == "__main__":
    test_tts()
