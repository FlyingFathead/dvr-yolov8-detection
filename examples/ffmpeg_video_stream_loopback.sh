#!/bin/bash

# OBS Studio can output its stream using various protocols. You can configure OBS to stream over RTMP locally, and then use FFmpeg to capture this stream. Here’s how to set it up:

#     Configure OBS:
#         Go to Settings -> Stream.
#         Set “Service” to “Custom…”.
#         Set the “Server” to something like rtmp://localhost/live. You'll need to set up an RTMP server on your localhost for this.

# can also be `localhost` if need be
# ffmpeg -listen 1 -f flv -i rtmp://localhost/live -c copy -f flv rtmp://127.0.0.1/liveout

# echo "Starting ffmpeg loopback RTMP server on 127.0.0.1 ..." 

# flv version
# ffmpeg -listen 1 -f flv -i rtmp://127.0.0.1/live -c copy -f flv rtmp://127.0.0.1/liveout

# ffmpeg -listen 1 -i rtmp://127.0.0.1/live -c:v libx264 -c:a aac -f mp4 rtmp://127.0.0.1/liveout
# ffmpeg -listen 1 -i rtmp://127.0.0.1/live/stream -c:v libx264 -c:a aac -f flv rtmp://127.0.0.1/liveout/stream

echo "Starting ffmpeg loopback RTMP server on 127.0.0.1:1935 ..."

# ffmpeg -listen 1 -i rtmp://127.0.0.1:1935/live -c:v libx264 -c:a aac -f flv rtmp://127.0.0.1:1935/liveout
# ffmpeg -listen 1 -i rtmp://127.0.0.1:1935/live -c:v libx264 -c:a aac -f flv rtmp://127.0.0.1:1935/liveout
# ffmpeg -v debug -listen 1 -i rtmp://127.0.0.1:1935/live -f flv rtmp://127.0.0.1:1935/liveout
# ffmpeg -v debug -listen 1 -i rtmp://127.0.0.1:1935/live -c:v libx264 -c:a aac -f flv rtmp://127.0.0.1:1936/liveout

ffmpeg -v debug -listen 1 -i rtmp://127.0.0.1:1935/live -c:v libx264 -c:a aac -f flv rtmp://127.0.0.1:1936/liveout