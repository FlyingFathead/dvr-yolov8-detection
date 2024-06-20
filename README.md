# dvr-yolo8-detection

- Real-time human / animal / object detection and alert system
- Uses Python + YOLOv8 + OpenCV2
- Supports CUDA GPU acceleration
- Separate batch processing tool for offline video file detections

## Overview

This project leverages YOLOv8 and OpenCV2 to perform object detection on either real-time video streams or batches of video files. It processes each video frame by frame, detects humans (by default; other specified objects can be added in as well), and logs the detections. Additionally, it saves frames with detected objects/entities highlighted.

CUDA enabled OpenCV2 is recommended for faster operation. **Note: CUDA-enabled OpenCV2 needs to be compiled manually and installed separately and the compiling process is highly dependent on your hardware setup, so I will not post separate guides on the compile process here.**

The real-time detection also supports additional CUDA features such as CUDA video denoising (note: requires CUDA compiled from source).

## Features

- Real-time object detection from RTMP streams
   - The source can be any RTMP stream (i.e. an IP camera, DVR that streams RTMP video, etc.)
   - Use any USB webcam/video feed as a source w/ OSB Studio + RTMP loopback
- Audio alerts for real-time detections over TTS using `pyttsx3`   
- Batch processing mode for video files for offline object/human/animal detection.

- Configurable via `config.ini` configuration file:
   - Customizable confidence threshold for detections.
   - Option to skip rescaling of video frames.
   - Option to log detection details into a separate log file.
   - Saves frames with detected objects as images.
   - Model variant selection (i.e. YOLOv8n, YOLOv8s, YOLOv8m, etc.)
   - Plenty of other options tho choose from...

## Requirements

- Python 3.6+
- torch
- ultralytics (YOLO)
- OpenCV (CUDA-enabled recommended for faster processing)
- numpy
- pyttsx3 (for text-to-speech alerts)
- ffmpeg (for handling RTMP streams)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/FlyingFathead/dvr-yolov8-detection.git
   cd dvr-yolov8-detection
   ```

2. Install the required Python packages:

   ```bash
   pip install torch ultralytics opencv-python-headless numpy pyttsx3 configparser
   ```

3. Install OpenCV:
   - For a headless version (no GUI support):
     ```bash
     pip install opencv-python-headless
     ```
   - For a full version (with GUI support):
     ```bash
     pip install opencv-python
     ```
   - **For CUDA-enabled OpenCV**, you need to build it from source. Follow the instructions on the [OpenCV documentation](https://docs.opencv.org/master/d6/d15/tutorial_building_tegra_cuda.html).

4. Install FFmpeg:
   - On Ubuntu:
     ```bash
     sudo apt-get install ffmpeg
     ```
   - On Windows and macOS, follow the instructions on the [FFmpeg download page](https://ffmpeg.org/download.html).

## Real-Time RTMP Stream Detection

This project also supports real-time object detection from an RTMP stream using YOLOv8. The provided `run_detection.sh` script ensures the detection script runs continuously, automatically restarting if it exits.

### Usage

1. **Ensure your RTMP server is set up and streaming:**
   - The example configuration (`example-nginx.conf`) can be used to set up an RTMP server with NGINX, for use cases where you need to redirect i.e. an USB webcam (or literally any video source using i.e. OBS Studio) to run the detection in real-time.   

2. **Run the detection script:**

   ```bash
   ./run_detection.sh
   ```

The `run_detection.sh` bash script will continuously run `yolov8_live_rtmp_stream_detection.py`, restarting it automatically if it exits.

### Configuration

You can configure various parameters by editing the `config.ini` file, this allows you to set program parameters such as the input stream address, output directory, confidence threshold, model variant, stream URL, and more.

### Example NGINX Configuration

An example NGINX configuration is provided in `example-nginx.conf`. This config sets up an RTMP server that listens on `127.0.0.1:1935` and allows local clients to publish and play streams. This is especially useful if you have a video source that's not internet-connected, such as any sort of USB camera or other kind of video feed that you can connect to i.e. OBS Studio. You can then use the local loopback via i.e. OBS Studio to get the video stream.

### RTMP Loopback Script

To stream and process the video in real-time, the `ffmpeg_video_stream_loopback.sh` script can be used. If using a streaming suite such as OBS Studio, ensure that your OBS Studio or streaming client is set to stream to `rtmp://127.0.0.1:1935/live`.

# Offline Batch Detection

You can use i.e. the `utils/batch_humdet_yolo8_opencv2.py` to run YOLOv8 batch detection on video files, suitable for offline use if you need to go through pre-existing video files.

## Changes
- v0.1402 - detection logging into a file added
- v0.1401 - configparser added; now configurable via `config.ini`
- v0.140 - more double-checking as to configuration options and their availability

## Credits
Developed by [FlyingFathead](https://github.com/FlyingFathead), with the usual digital ghost code from ChaosWhisperer.

## License
If you like it, let me know. :-)