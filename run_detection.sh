#!/bin/bash

# Path to the YOLOv8 RTMP stream detection script
SCRIPT_PATH="./yolov8_live_rtmp_stream_detection.py"

# Default environment type
ENV_TYPE="conda"  # Options: conda, virtualenv, none

# Name of the Conda environment
CONDA_ENV_NAME="yolov8_env"

# Function to check if Conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to activate Conda environment
activate_conda_env() {
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
}

# Function to check if a virtual environment is active
check_venv() {
    if [ -z "$VIRTUAL_ENV" ]; then
        return 1  # Virtual environment is not active
    else
        return 0  # Virtual environment is active
    fi
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --env)
        ENV_TYPE="$2"
        shift # past argument
        shift # past value
        ;;
        *)
        shift # past unrecognized argument
        ;;
    esac
done

# Activate the environment based on ENV_TYPE
case $ENV_TYPE in
    conda)
    if check_conda; then
        echo "Activating Conda environment '$CONDA_ENV_NAME'."
        activate_conda_env
    else
        echo "Conda not found. Exiting."
        exit 1
    fi
    ;;
    virtualenv)
    if check_venv; then
        echo "Using the active virtual environment at '$VIRTUAL_ENV'."
    else
        echo "No virtual environment is active. Exiting."
        exit 1
    fi
    ;;
    none)
    echo "Proceeding without activating any environment."
    ;;
    *)
    echo "Unknown environment type '$ENV_TYPE'. Exiting."
    exit 1
    ;;
esac

# Function to handle script termination
terminate_script() {
    echo "Termination signal received. Exiting..."
    exit 0
}

# Trap termination signals
trap terminate_script SIGINT SIGTERM

# Function to run the script
run_script() {
    while true; do
        echo "Starting YOLOv8 RTMP Stream Detection script..."
        python "$SCRIPT_PATH"

        # Check the exit code
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Script exited normally."
            # Break the loop if you don't want to restart on normal exit
            break
        else
            echo "Script crashed with exit code $EXIT_CODE. Restarting..."
        fi

        # Optional: Add a delay before restarting
        sleep 2
    done
}

# Run the script
run_script
