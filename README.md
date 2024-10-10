# DVR-YOLOv8-Detection

## Description

`dvr-yolov8-detection` is designed for real-time detection of humans, animals, or objects using the YOLOv8 model and OpenCV. It supports real-time video streams via RTMP or USB webcams, includes CUDA GPU acceleration for enhanced performance, and provides options for saving detections and logging events.

## Features

- **Real-time human/animal/object detection and alert system**
- Runs on **Python + YOLOv8 + OpenCV2**
- **CUDA GPU acceleration**; also runs in CPU-only mode
- Supports either **RTMP streams** or **USB webcams** for real-time video sources
  - Includes a loopback example and NGINX configuration example for RTMP use (i.e. OBS Studio)
- Detections can be automatically saved as images with a detection log
- Separate tool included for **offline video file detections** (see: `utils/`)

## Overview

The project uses Python with YOLOv8 and OpenCV2 to perform object detection on either real-time video streams or batches of video files. It processes each video frame by frame, detecting humans by default (other YOLOv8-supported objects can be added as needed). The program can log detections into a separate log file, save detection frames with highlighted objects, and send out audio alerts via `pyttsx3`.

Configuration options are available in the `config.ini` file.

**Note:** Using CUDA-enabled OpenCV is recommended for faster operation. CUDA-enabled OpenCV2 needs to be compiled manually and installed separately, as the compiling process is highly dependent on your hardware setup. Refer to the "Troubleshooting" section for guidance and an example build script for OpenCV with CUDA.

Real-time detection also supports additional CUDA features such as CUDA video denoising (note: this feature requires CUDA and is often available only when OpenCV is compiled from source).

### Configurable Features via `config.ini`:

- Video source (USB webcam or RTMP stream)
- Confidence threshold for detections
- Enable or disable rescaling of video frames
- CUDA-based video denoising (experimental)
- Log detection details into a separate log file
- Save frames with detected objects as image files
- Model variant selection (e.g., YOLOv8n, YOLOv8s, YOLOv8m)
- _... and other additional customizable options_

---

## 🐳 Dockerized Setup 🐳

Setting up `dvr-yolov8-detection` using Docker streamlines the deployment process, ensuring a consistent environment across different systems. This section guides you through building and running the Docker container, leveraging GPU acceleration for optimal performance.

### 🔧 **Prerequisites**

Before proceeding, ensure the following are installed and configured on your system:

1. **Docker Engine**
   - **Installation Guide:** [Docker Installation](https://docs.docker.com/engine/install/)

2. **NVIDIA Drivers**
   - **Ensure** that your system has the latest NVIDIA drivers installed to support CUDA GPU acceleration.
   - **Installation Guide:** [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

3. **NVIDIA Container Toolkit**
   - Enables Docker to access the GPU resources on the host machine.
   - **Installation Steps:**
     ```bash
     # Add the package repositories
     distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
     curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
     curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
     
     # Install the NVIDIA Container Toolkit
     sudo apt-get update
     sudo apt-get install -y nvidia-docker2
     
     # Restart the Docker daemon to apply changes
     sudo systemctl restart docker
     ```
   - **Reference:** [NVIDIA Docker Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

4. **GitHub Repository Clone**
   - Clone the `dvr-yolov8-detection` repository to your local machine:
     ```bash
     git clone https://github.com/FlyingFathead/dvr-yolov8-detection.git
     cd dvr-yolov8-detection
     ```

### 🛠️ **Building the Docker Image**

1. **Navigate to the Project Directory:**
   Ensure you're in the root directory of the cloned repository where the `Dockerfile` is located.
   ```bash
   cd dvr-yolov8-detection
   ```

2. **Build the Docker Image:**
   Execute the following command to build the Docker image named `yolov8_detection:latest`:
   ```bash
   docker build -t yolov8_detection:latest .
   ```
   - **Explanation:**
     - `docker build`: Command to build a Docker image from a Dockerfile.
     - `-t yolov8_detection:latest`: Tags the image with the name `yolov8_detection` and the tag `latest`.
     - `.`: Specifies the current directory as the build context.

   - **Note:** The build process may take some time, especially when compiling OpenCV with CUDA support.

### 🚀 **Running the Docker Container**

You have two options to run the Docker container: **Manually** or using the provided **`run_dockerized.sh` script**.

#### **Option 1: Manual Execution**

1. **Run the Container:**
   ```bash
   docker run --gpus all --network=host --rm yolov8_detection:latest
   ```
   - **Flags Explained:**
     - `--gpus all`: Grants the container access to all available GPUs.
     - `--network=host`: Shares the host's network stack with the container, allowing seamless access to services like RTMP servers running on `localhost`.
     - `--rm`: Automatically removes the container once it stops, keeping your system clean.

2. **Mount Configuration and Output Directories (Optional):**
   To customize configurations or persist detection outputs outside the container, use volume mounts:
   
   ```bash
   docker run --gpus all --network=host --rm \
     -v ./config.ini:/app/config.ini \
     -v ./yolo_detections:/app/yolo_detections \
     -v ./logs:/app/logs \
     yolov8_detection:latest
   ```
   - **Flags Explained:**
     - `-v ./config.ini:/app/config.ini`: Mounts your custom `config.ini` into the container.
     - `-v ./yolo_detections:/app/yolo_detections`: Persists detection images to the host.
     - `-v ./logs:/app/logs`: Stores log files on the host for easy access.

#### **Option 2: Using the `run_dockerized.sh` Script**

The `run_dockerized.sh` script automates the process of checking prerequisites and running the Docker container with the appropriate configurations.

1. **Ensure the Script is Executable:**
   ```bash
   chmod +x run_dockerized.sh
   ```

2. **Run the Script:**
   ```bash
   ./run_dockerized.sh
   ```
   - **Script Functionality:**
     - **Docker Installation Check:** Verifies if Docker is installed.
     - **Docker Service Status Check:** Ensures the Docker daemon is running.
     - **User Permissions Check:** Determines if the current user is part of the `docker` group to run Docker commands without `sudo`.
     - **Docker Image Availability Check:** Checks if the `yolov8_detection:latest` image exists locally; prompts to build it if not found.
     - **Container Execution:** Runs the Docker container with GPU access and host networking.

   - **Advantages:**
     - **Automated Checks:** Reduces manual verification steps.
     - **User-Friendly:** Provides clear messages and prompts for user actions.
     - **Enhanced Security:** Handles permission nuances seamlessly.

### ⚙️ **Configuration**

`dvr-yolov8-detection` utilizes a `config.ini` file to manage various operational parameters. Here's how to integrate it with your Docker setup:

1. **Locate or Create `config.ini`:**
   - If you have a custom `config.ini`, ensure it's prepared with your desired settings.
   - If not, refer to the default configuration provided in the repository and modify as needed.

2. **Mount `config.ini` into the Container:**
   ```bash
   docker run --gpus all --network=host --rm \
     -v ./config.ini:/app/config.ini \
     yolov8_detection:latest
   ```
   - **Benefit:** Allows you to modify configurations without rebuilding the Docker image.

3. **Persist Detection Outputs and Logs:**
   To ensure that detection results and logs are stored outside the container for later review:
   ```bash
   docker run --gpus all --network=host --rm \
     -v ./config.ini:/app/config.ini \
     -v ./yolo_detections:/app/yolo_detections \
     -v ./logs:/app/logs \
     yolov8_detection:latest
   ```
   - **Explanation:**
     - **Detections Volume (`/app/yolo_detections`):** Stores images with detected objects.
     - **Logs Volume (`/app/logs`):** Contains log files detailing detection events and system messages.

### 🗃️ **Data Persistence and Volume Management**

To maintain data across container restarts or to access outputs directly on your host system, utilize Docker volumes effectively.

1. **Create Host Directories:**
   ```bash
   mkdir -p ./yolo_detections
   mkdir -p ./logs
   ```
   
2. **Run the Container with Volume Mounts:**
   ```bash
   docker run --gpus all --network=host --rm \
     -v ./config.ini:/app/config.ini \
     -v ./yolo_detections:/app/yolo_detections \
     -v ./logs:/app/logs \
     yolov8_detection:latest
   ```
   - **Benefits:**
     - **Persistent Storage:** Data remains intact even if the container is removed.
     - **Easy Access:** Access detection images and logs directly from your host machine.

### 🐞 **Troubleshooting Tips**

- **No GPU Detected:**
  - Ensure that NVIDIA drivers and the NVIDIA Container Toolkit are correctly installed.
  - Verify Docker has access to the GPU by running:
    ```bash
    docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
    ```
    - **Expected Output:** Displays your GPU details.

- **RTMP Stream Issues:**
  - Confirm that your RTMP server is running and accessible.
  - Ensure the stream URL in `config.ini` matches your RTMP server settings.

- **Webcam Access Problems:**
  - Verify that the container has access to the webcam device.
  - You might need to specify device permissions or use privileged mode (use cautiously):
    ```bash
    docker run --gpus all --network=host --rm \
      --device=/dev/video0:/dev/video0 \
      yolov8_detection:latest
    ```

- **Permission Denied Errors:**
  - Ensure your user is part of the `docker` group.
  - Re-run the `run_dockerized.sh` script or use `sudo` if necessary.

- **Missing `requirements.txt`:**
  - Ensure that `requirements.txt` is present in the build context when building the Docker image.
  - Verify the `Dockerfile` has the correct `COPY` path for `requirements.txt`.

### 📝 **Additional Notes**

- **Host Networking Limitations:**
  - The `--network=host` flag is **only supported on Linux**. If you're using Docker Desktop on Windows or macOS, consider alternative networking configurations, such as using `host.docker.internal` or Docker Compose with a shared network.

- **Configuration Flexibility:**
  - By mounting the `config.ini` file, you can easily switch between different configurations without modifying the Docker image.

- **Automated Scripts:**
  - Utilize the provided `run_dockerized.sh` script for streamlined execution and to handle common setup checks automatically.

By following this Dockerized setup, you can efficiently deploy and manage the `dvr-yolov8-detection` application, leveraging the power of containerization and GPU acceleration for real-time detection tasks.

---

## Install Method B: Manual Setup

## Requirements

- **Python 3.6+** (Python 3.10.x recommended)
  - **Modules:**
    - `torch`
    - `ultralytics` (YOLO)
    - `opencv-python` (CUDA-enabled recommended for faster processing)
    - `numpy`
    - `pyttsx3` (for text-to-speech alerts)
    - `ffmpeg-python` (for video streams)
    - `pytz`

- **FFmpeg** 
- **Python 3.10.x**
- If you wish to use CUDA GPU acceleration, you will need:
  - A Nvidia GPU that supports CUDA
  - Install **CUDA 11.8** or higher to enable GPU-accelerated processing
- Use **Miniconda** or **Mamba** for environment management

## Installation (Conda/Mamba Environments)

1. **Clone the repository:**

   ```bash
   git clone https://github.com/FlyingFathead/dvr-yolov8-detection.git
   cd dvr-yolov8-detection
   ```

1.2 **(Install Miniconda or Anaconda if not already installed):**

- **Download and install Miniconda (recommended):**

  - For Linux/macOS:

    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    ```

  - For Windows, download the installer from [here](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions.

2. **Set up the environment Conda/Mamba environment:**

   ```bash
   ./setup_mamba_env.sh
   ```

   *This script creates a Conda/Mamba environment with the required dependencies.*

3. **Run the detection script:**

   ```bash
   ./run_detection.sh
   ```

## Installation (Manual Steps)

1. **Clone the repository:**

   ```bash
   git clone https://github.com/FlyingFathead/dvr-yolov8-detection.git
   cd dvr-yolov8-detection
   ```

2. **Install the required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *This ensures that all dependencies are managed through the `requirements.txt` file.*

3. **Install OpenCV:**

   - For a full version with GUI support (**recommended**):

     ```bash
     pip install opencv-python
     ```

   - **For CUDA-enabled OpenCV**, you need to build it from source. Refer to the [OpenCV documentation](https://docs.opencv.org/master/d6/d15/tutorial_building_tegra_cuda.html) or check the [build script](utils/install_and_compile_opencv_with_cuda.sh) for guidance.

4. **Install FFmpeg:**

   - On Ubuntu:

     ```bash
     sudo apt-get install ffmpeg
     ```

   - On Windows and macOS, follow the instructions on the [FFmpeg download page](https://ffmpeg.org/download.html).

## Real-Time Detection

This project supports real-time object detection from RTMP streams or USB webcams using YOLOv8. The provided `run_detection.sh` script ensures the detection script runs continuously, automatically restarting if it exits.

### Usage

#### **1. For RTMP Streams:**

- **Set up your RTMP server:**

  - Use the example NGINX configuration (`example-nginx.conf`) to set up an RTMP loopback server.
  - Ensure your streaming client (e.g., OBS Studio) is set to stream to `rtmp://127.0.0.1:1935/live`.

- **Run the detection script:**

  ```bash
  ./run_detection.sh
  ```

#### **2. For USB Webcams:**

- **Run the detection script with the `--use_webcam` option:**

  ```bash
  python3 yolov8_live_rtmp_stream_detection.py --use_webcam true
  ```

- **Specify webcam index (if needed):**

  ```bash
  python3 yolov8_live_rtmp_stream_detection.py --use_webcam true --webcam_index <number>
  ```

  - Replace `<number>` with the index number of your webcam.
  - Use the `utils/get_webcams.py` tool to find available webcams and their index numbers.

- **Alternatively, configure via `config.ini`:**

  - Set `use_webcam` to `true`.
  - Set `webcam_index` to your desired webcam index.

### Program Configuration

Configure various parameters by editing the `config.ini` file. This allows you to set program parameters such as the input source, input stream address, output directory, confidence threshold, model variant, stream URL, and more.

### Example NGINX Configuration

An example NGINX configuration is provided in `example-nginx.conf`. This config sets up an RTMP server that listens on `127.0.0.1:1935` and allows local clients to publish and play streams.

### RTMP Loopback Script

To stream and process the video in real-time, use the `ffmpeg_video_stream_loopback.sh` script. Ensure your streaming client (e.g., OBS Studio) is set to stream to `rtmp://127.0.0.1:1935/live`.

### Windows Users / Platform-Agnostic Loopback for RTMP

Use the `utils/loopback_test_unit_ffmpeg-python.py` script to set up a loopback for your RTMP stream.

1. **Install `ffmpeg-python`:**

   ```bash
   pip install -U ffmpeg-python
   ```

2. **Run the loopback script:**

   ```bash
   python3 utils/loopback_test_unit_ffmpeg-python.py
   ```

3. **Run the detection script:**

   ```bash
   python3 yolov8_live_rtmp_stream_detection.py
   ```

4. **Configure your streaming client to stream to:**

   ```
   rtmp://127.0.0.1:1935/live
   ```

**Note:** Using NGINX as a loopback method is highly recommended for stability.

## Offline Batch Detection Utility

Use `utils/batch_humdet_yolo8_opencv2.py` to run YOLOv8 batch detection on directories of video files, suitable for faster offline use.

## Troubleshooting

### RTMP Loopback Issues

- If for whatever reason the loopback doesn't seem working, you can create a test stream with i.e. `utils/test_stream_generator.py`. When ran, the script generates synthetic video frames and streams them to your RTMP server using FFmpeg as a subprocess, enabling you to try out if your loopback works. 
- Run the `test_stream_generator.py` and keep it running in the background, then try to first use VLC to connect to your stream (`VLC: Media -> Open Network Stream -> rtmp://127.0.0.1:1935/live/stream`). If this works, the main detection script should work as well.

### CUDA Not Found

- Ensure that you have all necessary modules installed with CUDA enabled.
- You may need to compile OpenCV from source to enable CUDA support.
- Refer i.e. to the [OpenCV w/ CUDA build script for Ubuntu 22.04LTS](examples/install_and_compile_opencv_with_cuda.sh) or [the 24.04LTS build script](examples/compile-opencv-with-cuda-ubuntu-24-04-lts.sh) for some degree of guidance.
- Verify CUDA support by checking if the program detects your GPU on startup.

### CUDA Installation

- See: [CUDA Setup Guide](CUDA_SETUP.md)

### Running Without a CUDA-Capable GPU

- The program can run in CPU-only mode, though performance may be slower.
- To improve performance:
  - Use a smaller model size in the `config.ini`.
  - Adjust rescale and frame rate options.
  - Reduce the resolution and frame rate from the video source.

## TODO

- Implement more error handling for edge cases
- Refactor code for improved modularity
- Add setup scripts for easier deployment
- Implement threshold settings for alerts (e.g., number of detections within a time frame)
- Add hooks for sending detections to web servers or APIs

## Changelog

- (Oct-10-2024) 🐳 **Dockerized Setup Now Available!** 🐳
  - Added Docker as an install method to ease the setup process
  - _(no other version changes)_
- **v0.153**
  - `config.ini` & program changes:
  - Fallback directory (`fallback_save_dir`)
  - Option to create date-based sub-directories (i.e. `/yolo_detections_path/year/month/day/`)
- **v0.152**
  - Added Conda/Mamba installer script for easier deployment
- **v0.151**
  - Added fallbacks for directories
- **v0.1501**
  - Fallback to non-CUDA modes if CUDA not supported
- **v0.15**
  - Added direct USB webcam support
    - Configure via `config.ini` or use `--use_webcam` flag
    - Added `utils/get_webcams.py` to find webcam indices
- **v0.1402**
  - Added detection logging to a file
- **v0.1401**
  - Added `configparser`; now configurable via `config.ini`
- **v0.140**
  - Improved configuration option checks

## Licensing

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.

### Ethos and Intent

I created this project to support non-profit and educational endeavors. While the GPLv3 license permits commercial use, I kindly ask that if you plan to use this project for commercial purposes, you consider reaching out to me. Your support and collaboration are greatly appreciated.

### Contact

For inquiries, suggestions, or collaborations, please contact me at `flyingfathead@protonmail.com` or visit [FlyingFathead on GitHub](https://github.com/FlyingFathead).

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests on GitHub, or contact the author (me) directly at `flyingfathead@protonmail.com`.

## Credits

Developed by [FlyingFathead](https://github.com/FlyingFathead), with digital ghost code contributions from ChaosWhisperer.

## Other

Star it if you like it. *;-)