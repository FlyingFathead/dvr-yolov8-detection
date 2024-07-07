# loopback_test_unit_ffmpeg-python.py
#
# This is a loopback test for use cases where you don't have access to i.e. Nginx
# It requires `ffmpeg-python`, please install with: `pip install ffmpeg-python`
#

import ffmpeg
import sys

def create_rtmp_server(source_url, output_url):
    try:
        # The process pulls from a source RTMP URL and pushes to a local RTMP URL
        stream = (
            ffmpeg
            .input(source_url)
            .output(output_url, format='flv', vcodec='copy', acodec='copy')
            .global_args('-re')  # Read input at native frame rate
        )
        ffmpeg.run(stream, overwrite_output=True)
    except ffmpeg.Error as e:
        print('Error:', e.stderr.decode(), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    source_url = 'rtmp://127.0.0.1:1935/live'
    output_url = 'rtmp://127.0.0.1:1935/live/stream'
    create_rtmp_server(source_url, output_url)
