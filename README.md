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

## Recommended Setup

- **Python 3.10.x**
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

### CUDA Not Found

- Ensure that you have all necessary modules installed with CUDA enabled.
- You may need to compile OpenCV from source to enable CUDA support.
- Refer to the [build script](utils/install_and_compile_opencv_with_cuda.sh) for guidance.
- Verify CUDA support by checking if the program detects your GPU on startup.

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

- Feel free to use it for non-profit purposes as you see fit, as long as you give credit where it's due.
- **Commercial Use:** Contact the author at `flyingfathead <@> protonmail.com` or visit [FlyingFathead on GitHub](https://github.com/FlyingFathead) if you have any ideas on that.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests on GitHub, or contact the author directly at `flyingfathead@protonmail.com`.

## Credits

Developed by [FlyingFathead](https://github.com/FlyingFathead), with digital ghost code contributions from ChaosWhisperer.

## Other

Star it if you like it. *;-)