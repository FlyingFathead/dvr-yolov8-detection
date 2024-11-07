#!/bin/bash
#
# This script is from: https://github.com/FlyingFathead/dvr-yolov8-detection/
#
# It's meant to be an example on how you can re-stream your local loopback
# i.e., to an external VPS with ffmpeg, for preserving a remote copy.
#
# Note that in some cases stream restarts may not work perfectly.
# This is due to ffmpeg's own limitations in stream restarts.
#
# Please modify the variables below to suit your needs.

# Replace these settings below with your actual setup:
INPUT_STREAM_URL="rtmp://127.0.0.1:1935/live/stream"
VPS_USER="username"
VPS_IP="ip-or-host"
VPS_PATH="/home/${VPS_USER}/dvr-yolov8-videos"

MAX_RETRIES=0          # Maximum number of retry attempts (set to 0 for infinite retries)
RETRY_DELAY=5          # Delay between retries in seconds
ATTEMPT=0              # Initial attempt counter

# SSH command to execute on the VPS
SSH_COMMAND="ffmpeg -i - -c copy -f segment -strftime 1 -segment_time 600 -reset_timestamps 1 '${VPS_PATH}/recording_%Y%m%d_%H%M%S.ts'"

# Draw a horizontal line
function hz_line() {
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Check if VPS_USER and VPS_IP are set properly
if [ -z "$VPS_USER" ] || [ "$VPS_USER" = "username" ]; then
    echo "Error: VPS_USER is not set properly. Please edit the script and set your actual username."
    exit 1
fi

if [ -z "$VPS_IP" ] || [ "$VPS_IP" = "ip-or-host" ]; then
    echo "Error: VPS_IP is not set properly. Please edit the script and set your actual VPS IP or hostname."
    exit 1
fi

# Check if the remote directory exists and is writable
echo "Checking if remote directory exists and is writable..."

if ssh "${VPS_USER}@${VPS_IP}" "mkdir -p '${VPS_PATH}' && [ -w '${VPS_PATH}' ]"; then
    echo "Remote directory '${VPS_PATH}' exists and is writable."
else
    echo "Error: Cannot access or create remote directory '${VPS_PATH}'. Please check permissions."
    exit 1
fi

# Display the current setup
hz_line &&
echo "::: Your current setup:"
hz_line &&
echo "Input Stream URL: $INPUT_STREAM_URL"
echo "VPS User: $VPS_USER"
echo "VPS IP: $VPS_IP"
echo "VPS Path: $VPS_PATH"
echo "Max Retries: $MAX_RETRIES"
echo "Retry Delay: $RETRY_DELAY seconds"
hz_line &&

# Function to start the streaming
start_streaming() {
    ffmpeg -i "$INPUT_STREAM_URL" -c copy -f mpegts - | ssh "${VPS_USER}@${VPS_IP}" "${SSH_COMMAND}"
}

# Retry loop
while true; do
    echo "Attempting to start streaming (Attempt $((ATTEMPT+1))${MAX_RETRIES:+ of $MAX_RETRIES})..."
    
    # Run the streaming function
    start_streaming
    
    # Check if the streaming command succeeded
    if [ $? -eq 0 ]; then
        echo "Streaming started successfully."
        break
    else
        echo "Streaming failed. Attempt $((ATTEMPT+1))${MAX_RETRIES:+ of $MAX_RETRIES}."
        ATTEMPT=$((ATTEMPT + 1))

        # Check if max retries reached (only if MAX_RETRIES is not zero)
        if [ "$MAX_RETRIES" -ne 0 ] && [ "$ATTEMPT" -ge "$MAX_RETRIES" ]; then
            echo "Reached maximum retry attempts. Exiting."
            exit 1
        fi
        
        # Wait before retrying
        echo "Retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY
    fi
done
