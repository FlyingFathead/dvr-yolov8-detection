# loopback_test_unit_ffmpeg-python.py
#
# This is a loopback test for use cases where you don't have access to i.e. Nginx
# It requires `ffmpeg-python`, please install with: `pip install ffmpeg-python`
#

import ffmpeg
import sys
import time
import logging

def create_rtmp_server(source_url, output_url):
    while True:
        try:
            logging.info(f"Attempting to connect to {source_url}")
            process = (
                ffmpeg
                .input(source_url, format='flv', listen=1)  # 'listen=1' makes ffmpeg wait for input
                .output(output_url, format='flv', vcodec='copy', acodec='copy')
                .global_args('-loglevel', 'info')  # Set loglevel to info
                .run_async(overwrite_output=True)
            )
            process.wait()
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
        
        logging.info("Waiting for stream to be available...")
        time.sleep(5)  # Wait for 5 seconds before retrying

if __name__ == "__main__":
    source_url = 'rtmp://127.0.0.1:1935/live'
    output_url = 'rtmp://127.0.0.1:1935/live/stream'

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting RTMP loopback server...")
    logging.info(f"Waiting for stream at {source_url} and forwarding to {output_url}")

    create_rtmp_server(source_url, output_url)
