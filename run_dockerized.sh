#!/bin/bash

# run_dockerized.sh
# A script to run the yolov8_detection Docker container with necessary checks.

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to display error messages
error_exit() {
    echo -e "\\033[0;31mERROR:\\033[0m $1" 1>&2
    exit 1
}

# Function to display info messages
info() {
    echo -e "\\033[0;32mINFO:\\033[0m $1"
}

# Function to display warning messages
warning() {
    echo -e "\\033[0;33mWARNING:\\033[0m $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    error_exit "Docker is not installed. Please install Docker and try again.
You can install Docker by following the instructions at https://docs.docker.com/get-docker/"
fi

info "Docker is installed."

# Check if Docker daemon is running
if ! sudo docker info > /dev/null 2>&1; then
    error_exit "Docker daemon is not running. Please start Docker and try again.
You can start Docker with: sudo systemctl start docker"
fi

info "Docker daemon is running."

# Check if the user can run Docker without sudo
if docker ps > /dev/null 2>&1; then
    DOCKER_CMD="docker"
else
    warning "Current user cannot run Docker commands without sudo."
    read -p "Do you want to run the Docker command with sudo? [y/N]: " USE_SUDO
    case "$USE_SUDO" in
        [yY][eE][sS]|[yY]) DOCKER_CMD="sudo docker" ;;
        *) error_exit "Docker command requires sudo. Exiting." ;;
    esac
fi

# Check if the Docker image exists locally
IMAGE_NAME="yolov8_detection:latest"
if ! $DOCKER_CMD image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    warning "Docker image '$IMAGE_NAME' not found locally."
    read -p "Do you want to build the Docker image now? [Y/n]: " BUILD_IMG
    BUILD_IMG=${BUILD_IMG:-Y}  # Default to 'Y' if empty
    case "$BUILD_IMG" in
        [yY][eE][sS]|[yY]|"")
            info "Building Docker image '$IMAGE_NAME'..."
            $DOCKER_CMD build -t "$IMAGE_NAME" .
            info "Docker image '$IMAGE_NAME' built successfully."
            ;;
        *)
            error_exit "Docker image '$IMAGE_NAME' is required. Exiting."
            ;;
    esac
else
    info "Docker image '$IMAGE_NAME' is available locally."
fi

# Check if the script is running on Linux (since --network=host is Linux-only)
OS_NAME=$(uname)
if [ "$OS_NAME" != "Linux" ]; then
    warning "--network=host is only supported on Linux. Adjusting network settings accordingly."
    # Optionally, you can set a different network or notify the user to modify the script
    # For simplicity, we'll proceed without --network=host
    NETWORK_OPTION=""
else
    NETWORK_OPTION="--network=host"
fi

# Check if GPUs are available (optional, requires nvidia-docker)
if command -v nvidia-smi &> /dev/null; then
    info "NVIDIA GPUs detected."
    GPU_OPTION="--gpus all"
else
    warning "No NVIDIA GPUs detected or NVIDIA drivers not installed."
    GPU_OPTION=""
fi

# Optional: Check if Nginx RTMP server is accessible
# Uncomment the following block if you want to perform this check
: <<'END'
NGINX_URL="rtmp://127.0.0.1:1935/live/stream"

# Function to check if the RTMP server is reachable
check_nginx_rtmp() {
    # Attempt to establish a TCP connection to the RTMP server
    nc -zv 127.0.0.1 1935 &> /dev/null
    if [ $? -ne 0 ]; then
        warning "Nginx RTMP server is not accessible at $NGINX_URL."
        read -p "Do you want to proceed anyway? [y/N]: " PROCEED
        case "$PROCEED" in
            [yY][eE][sS]|[yY]) ;;
            *) error_exit "Nginx RTMP server is required. Exiting." ;;
        esac
    else
        info "Nginx RTMP server is accessible at $NGINX_URL."
    fi
}

check_nginx_rtmp
END

# Run the Docker container
info "Starting Docker container '$IMAGE_NAME'..."
$DOCKER_CMD run $GPU_OPTION $NETWORK_OPTION

# run with:
# $DOCKER_CMD run $GPU_OPTION $NETWORK_OPTION --rm "$IMAGE_NAME"
# if you want to delete after running

info "Docker container '$IMAGE_NAME' has stopped."
