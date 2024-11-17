#!/bin/bash

# Path to the YOLOv8 RTMP stream detection script
SCRIPT_PATH="./yolov8_live_rtmp_stream_detection.py"
CMDLINE_OPTS="--headless --enable_webserver --webserver_host 127.0.0.1"
# CMDLINE_OPTS="--headless --enable_webserver --webserver_host 0.0.0.0"

# Default environment type
ENV_TYPE="conda"  # Options: conda, virtualenv, none

# Default SSH allowance
ALLOW_SSH=true  # Options: true, false

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

# Function to check if a virtual environment is active
check_venv() {
    if [ -z "$VIRTUAL_ENV" ]; then
        return 1  # Virtual environment is not active
    else
        return 0  # Virtual environment is active
    fi
}

# Initialize Conda
if check_conda; then
    echo "Initializing Conda..."
    # Initialize Conda for use in the script
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Conda not found. Exiting."
    exit 1
fi

# Function to display usage information
usage() {
    echo "Usage: $0 [--env <env_type>] [--allow-ssh <true|false>]"
    echo ""
    echo "Options:"
    echo "  --env        Specify the environment type. Options: conda, virtualenv, none. Default: conda"
    echo "  --allow-ssh  Allow SSH and SCP subprocesses within Firejail. Options: true, false. Default: true"
    echo "  -h, --help   Display this help message and exit."
    exit 0
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
        --allow-ssh)
        ALLOW_SSH="$2"
        # Validate the value
        if [[ "$ALLOW_SSH" != "true" && "$ALLOW_SSH" != "false" ]]; then
            echo "Error: --allow-ssh must be 'true' or 'false'."
            usage
        fi
        shift # past argument
        shift # past value
        ;;
        -h|--help)
        usage
        ;;
        *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

# Function to handle script termination
terminate_script() {
    echo "Termination signal received. Exiting..."
    exit 0
}

# Trap termination signals
trap terminate_script SIGINT SIGTERM

# Function to locate ssh and scp binaries and export their paths
locate_and_export_binaries() {
    if [ "$ALLOW_SSH" = true ]; then
        SSH_BIN=$(which ssh)
        SCP_BIN=$(which scp)

        if [ -z "$SSH_BIN" ]; then
            echo "Error: ssh binary not found. Please ensure SSH is installed."
            exit 1
        fi

        if [ -z "$SCP_BIN" ]; then
            echo "Error: scp binary not found. Please ensure SCP is installed."
            exit 1
        fi

        # Export the paths as environment variables
        export SSH_BIN
        export SCP_BIN

        echo "Located ssh at: $SSH_BIN"
        echo "Located scp at: $SCP_BIN"
        echo "NOTE: You will need to configure Firejail to allow these separately if you wish to use SSH & SCP remote sync."
    else
        echo "SSH and SCP subprocesses are NOT allowed within Firejail."
    fi
}

# Locate and export ssh and scp if allowed
locate_and_export_binaries

# Function to run the script
run_script() {
    while true; do
        echo "Starting YOLOv8 RTMP Stream Detection Start Script..."

        case $ENV_TYPE in
            conda)
                if check_conda; then
                    echo "Activating Conda environment '$CONDA_ENV_NAME'."
                    conda activate "$CONDA_ENV_NAME"

                    # Construct Firejail command
                    FIREJAIL_CMD=(firejail --noprofile)

                    if [ "$ALLOW_SSH" = true ]; then
                        FIREJAIL_CMD+=(--whitelist="$SSH_BIN" --whitelist="$SCP_BIN")
                        echo "SSH and SCP subprocesses are allowed within Firejail."
                    fi

                    # Append the Python command and options
                    FIREJAIL_CMD+=(python "$SCRIPT_PATH" $CMDLINE_OPTS)

                    # Execute the Firejail command
                    "${FIREJAIL_CMD[@]}"
                    EXIT_CODE=$?

                    # Deactivate the environment after the script finishes
                    conda deactivate
                else
                    echo "Conda not found. Exiting."
                    exit 1
                fi
                ;;
            virtualenv)
                if check_venv; then
                    echo "Using the active virtual environment at '$VIRTUAL_ENV'."

                    # Construct Firejail command
                    FIREJAIL_CMD=(firejail --noprofile)

                    if [ "$ALLOW_SSH" = true ]; then
                        FIREJAIL_CMD+=(--whitelist="$SSH_BIN" --whitelist="$SCP_BIN")
                        echo "SSH and SCP subprocesses are allowed within Firejail."
                    fi

                    # Append the Python command and options
                    FIREJAIL_CMD+=(python "$SCRIPT_PATH" $CMDLINE_OPTS)

                    # Execute the Firejail command
                    "${FIREJAIL_CMD[@]}"
                    EXIT_CODE=$?
                else
                    echo "No virtual environment is active. Exiting."
                    exit 1
                fi
                ;;
            none)
                echo "Proceeding without activating any environment."

                # Construct Firejail command
                FIREJAIL_CMD=(firejail)

                if [ "$ALLOW_SSH" = true ]; then
                    FIREJAIL_CMD+=(--whitelist="$SSH_BIN" --whitelist="$SCP_BIN")
                    echo "SSH and SCP subprocesses are allowed within Firejail."
                fi

                # Append the Python command and options
                FIREJAIL_CMD+=(python "$SCRIPT_PATH" $CMDLINE_OPTS)

                # Execute the Firejail command
                "${FIREJAIL_CMD[@]}"
                EXIT_CODE=$?
                ;;
            *)
                echo "Unknown environment type '$ENV_TYPE'. Exiting."
                exit 1
                ;;
        esac

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
