#!/usr/bin/env python3

import configparser
import subprocess
import sys
import os

def main():
    # Read config
    config = configparser.ConfigParser()
    files_read = config.read('config.ini')
    if not files_read:
        print("Could not load config.ini, aborting.")
        sys.exit(1)

    # Some example fields from config.ini you'd define 
    # (adjust naming to match your actual config)
    # or maybe you want a dedicated [hls] section:
    input_stream = config.get('stream', 'stream_url', fallback='rtmp://127.0.0.1:1935/live/stream')
    hls_dir = config.get('hls', 'hls_output_dir', fallback='/tmp/hls')
    hls_time = config.get('hls', 'hls_time', fallback='2')
    hls_list_size = config.get('hls', 'hls_list_size', fallback='10')
    segment_pattern = config.get('hls', 'segment_pattern', fallback='segment_%03d.ts')
    playlist_filename = config.get('hls', 'playlist_filename', fallback='playlist.m3u8')

    # Ensure the directory exists
    os.makedirs(hls_dir, exist_ok=True)

    # Build the full paths
    segment_path_pattern = os.path.join(hls_dir, segment_pattern)
    playlist_path = os.path.join(hls_dir, playlist_filename)

    # Construct the ffmpeg command
    command = [
        'ffmpeg',
        '-i', input_stream,
        '-c', 'copy',   # no re-encode
        '-f', 'hls',
        '-hls_time', str(hls_time),
        '-hls_list_size', str(hls_list_size),
        '-hls_segment_filename', segment_path_pattern,
        playlist_path
    ]

    print("Running FFmpeg command:")
    print(" ".join(command))

    # Run the process
    process = subprocess.Popen(command)
    try:
        process.wait()
    except KeyboardInterrupt:
        print("Received Ctrl+C, terminating ffmpeg...")
        process.terminate()

if __name__ == '__main__':
    main()
