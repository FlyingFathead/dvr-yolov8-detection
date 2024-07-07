# dvr-yolo8-detection

- Real-time human / animal / object detection and alert system
- Runs on Python + YOLOv8 + OpenCV2
- Supports RTMP streams for real-time video sources
   - (a loopback example + nginx config example included)
- Supports USB webcams for real-time video sources
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
     - The example configuration (`example-nginx.conf`) can be used to set up an RTMP loopback server with NGINX, this is practical for use cases where you need to redirect i.e. practically any video source using i.e. OBS Studio's streaming functionality to run the detection in real-time over RTMP via a local loopback.   

   - Run the detection script:

     ```bash
     ./run_detection.sh
     ```

   The `run_detection.sh` bash script will continuously run `yolov8_live_rtmp_stream_detection.py`, restarting it automatically if it exits.

2. **For USB Webcams:**

   This is probably the easier method if you have a USB webcam installed on your system and only need it as a source. You can also use OBS's Virtual Webcam as a source.

   - Run the main real-time detection script with the `--use_webcam` option:

     ```bash
     python3 yolov8_live_rtmp_stream_detection.py --use_webcam true
     ```

   Or, if you want to define your webcame source separately via its index number:

     ```bash
     python3 yolov8_live_rtmp_stream_detection.py --use_webcam true --webcam_index <number>
     ```

   - Replace `<number>` with the index number of your webcam. 
   - Use the `utils/get_webcams.py` tool to find the available webcams and their index numbers on your system if needed.
   - You can also adjust the webcam source as your default by editing the `config.ini` and setting the `use_webcam` flag to `true`
   - Set your webcam index number into the `webcam_index` configuration option

### Program Configuration

You can configure various parameters by editing the `config.ini` file, this allows you to set program parameters such as the input source (USB webcam, RTMP stream), input stream address, output directory, confidence threshold, model variant, stream URL, and much more.

### Example NGINX Configuration

An example NGINX configuration is provided in `example-nginx.conf`. This config sets up an RTMP server that listens on `127.0.0.1:1935` and allows local clients to publish and play streams. This is especially useful if you have a video source that's not internet-connected, such as any sort of USB camera or other kind of video feed that you can connect to i.e. OBS Studio. You can then use the local loopback via i.e. OBS Studio to get the video stream.

### RTMP Loopback Script

To stream and process the video in real-time, the `ffmpeg_video_stream_loopback.sh` script can be used. If using a streaming suite such as OBS Studio, ensure that your OBS Studio or streaming client is set to stream to `rtmp://127.0.0.1:1935/live`.

### Windows Users / Quick Platform-Agnostic Loopback for RTMP

You can run the `loopback_test_unit_ffmpeg-python.py` script located in the `utils/` directory to set up a loopback for your RTMP stream. If you're using this option: Make sure you have the `ffmpeg-python` module installed. You can install it with:

```bash
pip install -U ffmpeg-python
```

After you have the `ffmpeg-python` module installed:

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

## Offline Batch Detection Utility

You can use i.e. the `utils/batch_humdet_yolo8_opencv2.py` to run YOLOv8 batch detection on video file directories, suitable for faster offline use if you need to go through pre-existing video files.

## Changes

- v0.1501 - fallback to non-CUDA modes if CUDA not supported
- v0.15 - webcam support added!
   - you can edit the `config.ini` and set the input to webcam
   - or, run the main program with: `--use_webcam --webcam_index X` (where X is your webcam index)
   - added `utils/get_webcams.py` - a tool that you can run to check for available webcams and their index numbers
- v0.1402 - detection logging into a file added
- v0.1401 - configparser added; now configurable via `config.ini`
- v0.140 - more double-checking as to configuration options and their availability

## Credits
Developed by [FlyingFathead](https://github.com/FlyingFathead), with the usual digital ghost code from ChaosWhisperer.

## Troubleshooting

### CUDA not found
- Make sure that you have all the necessary modules installed with CUDA enabled. In many cases, enabling CUDA support might require you to compile OpenCV directly from source, which can be a daunting task especially for those with no prior experience. I have included my own OpenCV compile script [here](https://github.com/FlyingFathead/dvr-yolov8-detection/blob/main/utils/install_and_compile_opencv_with_cuda.sh), however I **do not recommend you run it** without understanding what it does. I have only used it to compile OpenCV with CUDA on a single Ubuntu Linux setup during summer 2024, so it's _not_ intended for everyone, and might already be outdated as you read this. Take what you will out of it at your own peril -- you have been warned.

- There are multiple other guides online for compiling OpenCV with CUDA support, each with varying degrees of success.

- The main real-time detection program does have a check on startup that prints out whether or not CUDA has been found and which GPU is in use, it can be useful for checking out your CUDA support status.

- Good luck! _(you'll need it, as well as patience)_

### What if I don't have a CUDA-capable GPU?
- You can still run the program in CPU-only mode, although it will be _extremely_ sluggish (don't expect too many frames per second). Should the model be too heavy to run on your computer, you can try changing it to a smaller one in the `config.ini` or even trying some of the rescale and frame rate options, although they are not always optimal either, as rescaling adds an extra step -- if possible, reduce the resolution and frame rate from your device's end, or, if you're using OBS Studio, from its output settings.

## Contributing

Contributions are always welcome! You can even leave your own development ideas either here on GitHub or directly via mail to flyingfathead <@> protonmail dot com.

## TODO

- More error catching in edge cases
- Refactoring and added modularity
- Setup scripts
- Threshold for sending out audio & other alerts 
   - (i.e.: X [number of] detections with Y confidence within Z seconds)
- Hooks for i.e. sending detections remotely to web servers, bot API's etc.

## Licensing

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. For non-commercial use, you are free to share and adapt the material with appropriate credit. For commercial use, please contact us to obtain a separate commercial license.

- **Non-Commercial Use**: Follow the terms of the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Commercial Use**: Contact the author at flyingfathead <@> protonmail.com or visit [https://github.com/FlyingFathead](https://github.com/FlyingFathead) to discuss licensing options.

---

If you like it, let me know. :-)