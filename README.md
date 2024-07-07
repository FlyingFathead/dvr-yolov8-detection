# dvr-yolo8-detection

- Real-time human / animal / object detection and alert system
- Runs on Python + YOLOv8 + OpenCV2
- Supports RTMP streams for real-time video processing sources
   - (a loopback example included for use with i.e. OBS Studio)
- Supports USB webcams for video source
- Supports CUDA GPU acceleration
- Detections can be automatically saved to images with a detection log
- Separate tool included for offline video file detections

## Overview

This project leverages YOLOv8 and OpenCV2 to perform object detection on either real-time video streams or batches of video files. It processes each video frame by frame, detects humans (by default; other specified objects can be added in as well), and logs the detections. Additionally, it saves frames with detected objects/entities highlighted.

CUDA enabled OpenCV2 is recommended for faster operation. **Note: CUDA-enabled OpenCV2 needs to be compiled manually and installed separately and the compiling process is highly dependent on your hardware setup, so I will not post separate guides on the compile process here.**

The real-time detection also supports additional CUDA features such as CUDA video denoising (note: requires CUDA compiled from source).

## Features

- Real-time object detection from RTMP streams or USB webcams
   - The source can be any RTMP stream (i.e. an IP camera, DVR that streams RTMP video, etc.)
   - Use any USB webcam as a source directly or via a RTMP loopback
- Audio alerts for real-time detections over TTS using `pyttsx3`   
- Batch processing mode for video files for offline object/human/animal detection.

- Configurable features via `config.ini`, i.e.:
   - Your USB webcam or RTMP video source
   - Confidence threshold for detections
   - Skip rescaling of video frames
   - Log detection details into a separate log file
   - Save frames with detected objects as image files
   - Model variant selection (i.e. YOLOv8n, YOLOv8s, YOLOv8m, etc.)
   - Plenty of other options to choose from...

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
   pip install -r requirements.txt
   ```

This approach ensures that the dependencies are managed through the `requirements.txt` file, making it easier to maintain and update them in the future.

- (You can also install the required Python packages manually if need be):

   ```bash
   pip install torch ultralytics opencv-python-headless numpy pyttsx3 configparser
   ```

3. Install OpenCV:
   - For a full version (with GUI support, **recommended**):
     ```bash
     pip install opencv-python
     ```
   - For a headless version (no GUI support, **not recommended unless using offline batch processing only**):
     ```bash
     pip install opencv-python-headless
     ```
   - **For CUDA-enabled OpenCV**, you need to build it from source. Follow the instructions on the [OpenCV documentation](https://docs.opencv.org/master/d6/d15/tutorial_building_tegra_cuda.html).

4. Install FFmpeg:
   - On Ubuntu:
     ```bash
     sudo apt-get install ffmpeg
     ```
   - On Windows and macOS, follow the instructions on the [FFmpeg download page](https://ffmpeg.org/download.html).

## Real-Time Detection

This project supports real-time object detection from RTMP streams or USB webcams using YOLOv8. The provided `run_detection.sh` script ensures the detection script runs continuously, automatically restarting if it exits.

### Usage

1. **For RTMP Streams:**
   - Ensure your RTMP server is set up and streaming:
     - The example configuration (`example-nginx.conf`) can be used to set up an RTMP server with NGINX, for use cases where you need to redirect i.e. a USB webcam (or literally any video source using i.e. OBS Studio) to run the detection in real-time.   

   - Run the detection script:

     ```bash
     ./run_detection.sh
     ```

   The `run_detection.sh` bash script will continuously run `yolov8_live_rtmp_stream_detection.py`, restarting it automatically if it exits.

2. **For USB Webcams:**
   - Run the detection script with the `--use_webcam` option:

     ```bash
     python3 yolov8_live_rtmp_stream_detection.py --use_webcam --webcam_index X
     ```

   Replace `X` with the index of your webcam. Use the `utils/get_webcams.py` tool to find the available webcams and their index numbers on your system if needed.

   You can also adjust the webcam source as your default by editing the `config.ini` and setting the `use_webcam` flag to `true` -- don't forget to add your appropriate webcam index to the `webcam_index` config flag!

### Configuration

You can configure various parameters by editing the `config.ini` file, this allows you to set program parameters such as the input stream address, output directory, confidence threshold, model variant, stream URL, and more.

### Example NGINX Configuration

An example NGINX configuration is provided in `example-nginx.conf`. This config sets up an RTMP server that listens on `127.0.0.1:1935` and allows local clients to publish and play streams. This is especially useful if you have a video source that's not internet-connected, such as any sort of USB camera or other kind of video feed that you can connect to i.e. OBS Studio. You can then use the local loopback via i.e. OBS Studio to get the video stream.

### RTMP Loopback Script

To stream and process the video in real-time, the `ffmpeg_video_stream_loopback.sh` script can be used. If using a streaming suite such as OBS Studio, ensure that your OBS Studio or streaming client is set to stream to `rtmp://127.0.0.1:1935/live`.

### Windows Users / Quick Platform-Agnostic Loopback

You can run the `loopback_test_unit_ffmpeg-python.py` script located in the `utils/` directory to set up a loopback for your RTMP stream.

#### Prerequisites
Make sure you have the `ffmpeg-python` module installed. You can install it with:

```bash
pip install -U ffmpeg-python
```

#### Running the Loopback Script
1. Run the loopback script in one terminal window:

   ```bash
   python3 utils/loopback_test_unit_ffmpeg-python.py
   ```

2. Leave that terminal window **active and running**.

3. Open another terminal window and run the main program:

   ```bash
   python3 yolov8_live_rtmp_stream_detection.py
   ```

4. Configure your OBS Studio's video stream output to:

   ```plaintext
   rtmp://127.0.0.1:1935/live
   ```

This setup will ensure that OBS streams to `rtmp://127.0.0.1:1935/live`, and the loopback script will forward it to `rtmp://127.0.0.1:1935/live/stream`, which your detection script will then process. However, using i.e. Nginx as a loopback method is highly recommended for stability.

# Offline Batch Detection

You can use i.e. the `utils/batch_humdet_yolo8_opencv2.py` to run YOLOv8 batch detection on video files, suitable for offline use if you need to go through pre-existing video files.

## Changes
- v0.15 - webcam support added!
   - you can edit the `config.ini` and set the input to webcam
   - or, run the main program with: `--use_webcam --webcam_index X` (where X is your webcam index)
   - added `utils/get_webcams.py` - a tool that you can run to check for available webcams and their index numbers
- v0.1402 - detection logging into a file added
- v0.1401 - configparser added; now configurable via `config.ini`
- v0.140 - more double-checking as to configuration options and their availability

## Credits
Developed by [FlyingFathead](https://github.com/FlyingFathead), with the usual digital ghost code from ChaosWhisperer.

## Licensing

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. For non-commercial use, you are free to share and adapt the material with appropriate credit. For commercial use, please contact us to obtain a separate commercial license.

- **Non-Commercial Use**: Follow the terms of the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Commercial Use**: Contact the author at flyingfathead <@> protonmail.com or visit [https://github.com/FlyingFathead](https://github.com/FlyingFathead) to discuss licensing options.

---

If you like it, let me know. :-)