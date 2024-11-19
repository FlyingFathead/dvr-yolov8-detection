#!/bin/bash

# =======================
# Configuration Variables
# =======================

SCREEN_NAME="stream-relay"  # Name for the screen session
RUN_IN_SCREEN=true          # Set to 'true' to run in screen
SUPERVISE_ERRORS=false      # Set to 'true' to enable error supervision/restart
MAX_RETRIES=0               # Max retries (0 = infinite)
RETRY_DELAY=5               # Delay between retries in seconds
REMUX_TO_MP4=false          # Toggle remux to MP4 or .ts
INPUT_STREAM_URL="rtmp://127.0.0.1:1935/live/stream"

VPS_USER="your-username"
VPS_IP="your-host"
VPS_PATH="/home/${VPS_USER}/your_dir"
LOG_FILE="/tmp/ffmpeg_watchdog.log"
MAX_LOG_SIZE=$((10 * 1024 * 1024)) # 10 MB in bytes

# Determine file extension
EXTENSION=$([[ "$REMUX_TO_MP4" == true ]] && echo "mp4" || echo "ts")

SSH_COMMAND="ffmpeg \
    -rw_timeout 5000000 \
    -fflags +genpts+igndts+discardcorrupt \
    -avoid_negative_ts make_zero \
    -i - \
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

RETRY_DELAY=${RETRY_DELAY:-5}
export RETRY_DELAY

ERROR_PATTERNS=("invalid dropping" "Broken pipe" "Conversion failed")

# =========
# Functions
# =========

# Manage log file size
manage_log_size() {
    if [ -f "${LOG_FILE}" ] && [ "$(stat -c%s "${LOG_FILE}")" -ge "${MAX_LOG_SIZE}" ]; then
        echo "Log file exceeded ${MAX_LOG_SIZE} bytes. Rotating..."
        mv "${LOG_FILE}" "${LOG_FILE}.old"
        : > "${LOG_FILE}" # Truncate the log file
    fi
}

# Start streaming logic
start_ffmpeg() {
    manage_log_size
    echo "Starting streaming from $INPUT_STREAM_URL to ${VPS_USER}@${VPS_IP}..."
    ffmpeg -i "$INPUT_STREAM_URL" -c copy -f mpegts - | ssh "${VPS_USER}@${VPS_IP}" "${SSH_COMMAND}" 2>&1 | tee -a "${LOG_FILE}" &
    FFMPEG_PID=$!
}

# Monitor FFmpeg for errors and restart if necessary
monitor_ffmpeg() {
    echo "Monitoring FFmpeg for errors..."
    tail -F "${LOG_FILE}" | while read -r line; do
        for pattern in "${ERROR_PATTERNS[@]}"; do
            if echo "$line" | grep -q "$pattern"; then
                echo "Detected error: $pattern. Restarting FFmpeg..."
                kill -9 "${FFMPEG_PID}" 2>/dev/null
                start_ffmpeg
            fi
        done
    done
}

# Retry streaming with a loop
stream_process() {
    ATTEMPT=0
    while true; do
        echo "Attempt $((ATTEMPT+1)) of ${MAX_RETRIES:-infinite}"
        
        # Validate SSH before starting streaming
        ssh "${VPS_USER}@${VPS_IP}" exit
        if [ $? -ne 0 ]; then
            echo "Error: Unable to connect to ${VPS_USER}@${VPS_IP}. Retrying in ${RETRY_DELAY} seconds..."
            sleep "${RETRY_DELAY}"
            continue
        fi

        # Start FFmpeg
        start_ffmpeg

        # Supervise errors if enabled
        if [[ "$SUPERVISE_ERRORS" == true ]]; then
            monitor_ffmpeg
        else
            wait "${FFMPEG_PID}"
        fi

        # Handle retries
        ((ATTEMPT++))
        if [[ "$MAX_RETRIES" -ne 0 && "$ATTEMPT" -ge "$MAX_RETRIES" ]]; then
            echo "Max retries reached. Exiting."
            exit 1
        fi
        sleep "${RETRY_DELAY}"
    done
}

# Handle screen session
run_in_screen() {
    if screen_exists; then
        echo "Screen session '${SCREEN_NAME}' already exists."
        echo "Attach using: screen -r ${SCREEN_NAME}"
    else
        echo "Starting screen session '${SCREEN_NAME}'..."
        
        # Export all necessary functions and variables to the screen session
        export -f stream_process
        export -f start_ffmpeg
        export -f monitor_ffmpeg
        export SSH_COMMAND INPUT_STREAM_URL VPS_USER VPS_IP VPS_PATH EXTENSION LOG_FILE ERROR_PATTERNS MAX_LOG_SIZE
        
        # Start the process inside a screen session
        screen -dmS "${SCREEN_NAME}" bash -c 'stream_process'
        echo "Attach to the screen session to view logs: screen -r ${SCREEN_NAME}"
    fi
}

# Check if a screen session exists
screen_exists() {
    screen -list | grep -q "\.${SCREEN_NAME}[[:space:]]"
}

# ===========
# Main Script
# ===========

# Check remote directories
ssh "${VPS_USER}@${VPS_IP}" "mkdir -p '${VPS_PATH}/hls_segments' && [ -w '${VPS_PATH}' ]" || {
    echo "Error: Remote directories not accessible. Check VPS permissions."
    exit 1
}

# Start streaming
if [[ "$RUN_IN_SCREEN" == true ]]; then
    run_in_screen
else
    stream_process
fi
