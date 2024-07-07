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
            # The process pulls from a source RTMP URL and pushes to a local RTMP URL
            (
                ffmpeg
                .input(source_url, format='flv')
                .output(output_url, format='flv', vcodec='copy', acodec='copy')
                .global_args('-loglevel', 'info')  # Set loglevel to info
                .run(overwrite_output=True)
            )
        except ffmpeg.Error as e:
            logging.error(f"Error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            time.sleep(5)  # Wait for 5 seconds before retrying
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            time.sleep(5)  # Wait for 5 seconds before retrying

if __name__ == "__main__":
    source_url = 'rtmp://127.0.0.1:1935/live'
    output_url = 'rtmp://127.0.0.1:1935/live/stream'

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting RTMP loopback server...")
    logging.info(f"Waiting for stream at {source_url} and forwarding to {output_url}")

    create_rtmp_server(source_url, output_url)
