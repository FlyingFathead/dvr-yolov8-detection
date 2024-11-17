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

# ============================
# Configuration Variables
# ============================

# Replace these settings below with your actual setup:
INPUT_STREAM_URL="rtmp://127.0.0.1:1935/live/stream"
VPS_USER="username"
VPS_IP="ip-or-host"
VPS_PATH="/home/${VPS_USER}/dvr-yolov8-videos"

# Screen configuration
SCREEN_NAME="dvr-stream-relay"  # Set your desired screen session name
RUN_IN_SCREEN=true          # Set to 'true' to run in screen, 'false' otherwise

# Streaming configuration
MAX_RETRIES=0               # Maximum number of retry attempts (set to 0 for infinite retries)
RETRY_DELAY=5               # Delay between retries in seconds
ATTEMPT=0                   # Initial attempt counter

# Toggle for remuxing to MP4
REMUX_TO_MP4=false          # Set to 'true' to remux to MP4, 'false' for direct .ts files
                            # MP4 is usually better supported by browsers compared to .ts
                            # Remuxing creates minimal overhead to the processing;
                            # it should not alter the quality otherwise.

# ============================
# SSH Command Configuration
# ============================

# Define extension based on REMUX_TO_MP4
if [ "$REMUX_TO_MP4" = true ]; then
    EXTENSION="mp4"
else
    EXTENSION="ts"
fi

# SSH Command Configuration
SSH_COMMAND="ffmpeg -i - \
    -c copy -map 0 \
    -f hls \
    -hls_time 2 \
    -hls_list_size 5 \
    -hls_flags delete_segments+append_list \
    -hls_allow_cache 0 \
    -hls_segment_filename '${VPS_PATH}/hls_segments/segment_%03d.ts' \
    '${VPS_PATH}/hls_segments/playlist.m3u8' \
    -c copy \
    -f segment \
    -strftime 1 \
    -segment_time 600 \
    -reset_timestamps 1 \
    '${VPS_PATH}/recording_%Y%m%d_%H%M%S.${EXTENSION}'"

# ============================
# Helper Functions
# ============================

# Draw a horizontal line
hz_line() {
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Check if a screen session exists
screen_exists() {
    screen -list | grep -q "\.${SCREEN_NAME}[[:space:]]"
}

# Start streaming function
start_streaming() {
    ffmpeg -i "$INPUT_STREAM_URL" -c copy -f mpegts - | ssh "${VPS_USER}@${VPS_IP}" "${SSH_COMMAND}"
}

# ============================
# Initial Checks
# ============================

# Check if VPS_USER and VPS_IP are set properly
if [ -z "$VPS_USER" ] || [ "$VPS_USER" = "username" ]; then
    echo "Error: VPS_USER is not set properly. Please edit the script and set your actual username."
    exit 1
fi

if [ -z "$VPS_IP" ] || [ "$VPS_IP" = "ip-or-host" ]; then
    echo "Error: VPS_IP is not set properly. Please edit the script and set your actual VPS IP or hostname."
    exit 1
fi

# Check if the remote directories exist and are writable
echo "Checking if remote directories exist and are writable..."

if ssh "${VPS_USER}@${VPS_IP}" "mkdir -p '${VPS_PATH}/hls_segments' && [ -w '${VPS_PATH}' ]"; then
    echo "Remote directories '${VPS_PATH}' and '${VPS_PATH}/hls_segments' exist and are writable."
else
    echo "Error: Cannot access or create remote directories. Please check permissions."
    exit 1
fi

# Display the current setup
hz_line
echo "::: Your current setup:"
hz_line
echo "Input Stream URL: $INPUT_STREAM_URL"
echo "VPS User: $VPS_USER"
echo "VPS IP: $VPS_IP"
echo "VPS Path: $VPS_PATH"
echo "Max Retries: $MAX_RETRIES"
echo "Retry Delay: $RETRY_DELAY seconds"
echo "Remux to MP4: $REMUX_TO_MP4"
hz_line

# ============================
# Streaming Logic
# ============================

# Define the main streaming process
stream_process() {
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
            sleep "$RETRY_DELAY"
        fi
    done
}

# ============================
# Screen Session Handling
# ============================

if [ "$RUN_IN_SCREEN" = true ]; then
    if screen_exists; then
        echo "A screen session named '${SCREEN_NAME}' already exists."
        echo "To attach to it, run: screen -r ${SCREEN_NAME}"
        exit 0
    else
        echo "Starting streaming in a new screen session named '${SCREEN_NAME}'..."
        # Start the screen session and set up the environment
        screen -dmS "${SCREEN_NAME}" bash -c "
            # Export variables
            VPS_USER='${VPS_USER}'
            VPS_IP='${VPS_IP}'
            VPS_PATH='${VPS_PATH}'
            INPUT_STREAM_URL='${INPUT_STREAM_URL}'
            REMUX_TO_MP4='${REMUX_TO_MP4}'
            MAX_RETRIES=${MAX_RETRIES}
            RETRY_DELAY=${RETRY_DELAY}

            # Define extension based on REMUX_TO_MP4
            if [ \"\$REMUX_TO_MP4\" = true ]; then
                EXTENSION=\"mp4\"
            else
                EXTENSION=\"ts\"
            fi

            # Reconstruct SSH_COMMAND inside the screen session
            SSH_COMMAND=\"ffmpeg -i - \
                -c copy -map 0 \
                -f hls \
                -hls_time 2 \
                -hls_list_size 5 \
                -hls_flags delete_segments+append_list \
                -hls_allow_cache 0 \
                -hls_segment_filename '\${VPS_PATH}/hls_segments/segment_%03d.ts' \
                '\${VPS_PATH}/hls_segments/playlist.m3u8' \
                -c copy \
                -f segment \
                -strftime 1 \
                -segment_time 600 \
                -reset_timestamps 1 \
                '\${VPS_PATH}/recording_%Y%m%d_%H%M%S.\${EXTENSION}'\"

            # Define functions
            hz_line() {
                printf '%*s\n' \"\${COLUMNS:-\$(tput cols)}\" '' | tr ' ' -
            }

            start_streaming() {
                ffmpeg -i \"\$INPUT_STREAM_URL\" -c copy -f mpegts - | ssh \"\${VPS_USER}@\${VPS_IP}\" \"\${SSH_COMMAND}\"
            }

            stream_process() {
                ATTEMPT=0
                while true; do
                    echo \"Attempting to start streaming (Attempt \$((ATTEMPT+1))${MAX_RETRIES:+ of \$MAX_RETRIES})...\"

                    # Run the streaming function
                    start_streaming

                    # Check if the streaming command succeeded
                    if [ \$? -eq 0 ]; then
                        echo \"Streaming started successfully.\"
                        break
                    else
                        echo \"Streaming failed. Attempt \$((ATTEMPT+1))${MAX_RETRIES:+ of \$MAX_RETRIES}.\"
                        ATTEMPT=\$((ATTEMPT + 1))

                        # Check if max retries reached (only if MAX_RETRIES is not zero)
                        if [ \"\$MAX_RETRIES\" -ne 0 ] && [ \"\$ATTEMPT\" -ge \"\$MAX_RETRIES\" ]; then
                            echo \"Reached maximum retry attempts. Exiting.\"
                            exit 1
                        fi

                        # Wait before retrying
                        echo \"Retrying in \$RETRY_DELAY seconds...\"
                        sleep \"\$RETRY_DELAY\"
                    fi
                done
            }

            # Start the streaming process
            stream_process

            # Keep the screen session alive
            exec bash
        "
        echo "Streaming has been started in screen session '${SCREEN_NAME}'."
    fi
else
    # Run the streaming process directly without screen
    stream_process
fi