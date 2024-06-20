#!/bin/bash

# Path to the YOLOv8 RTMP stream detection script
SCRIPT_PATH="./yolov8_live_rtmp_stream_detection.py"

# Function to run the script
run_script() {
    while true; do
        echo "Starting YOLOv8 RTMP Stream Detection script..."
        python3 $SCRIPT_PATH

        # Check the exit code
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Script exited normally. Restarting..."
        else
            echo "Script crashed with exit code $EXIT_CODE. Restarting..."
        fi

        # Optional: Add a delay before restarting
        sleep 2
    done
}

# Run the script
run_script
