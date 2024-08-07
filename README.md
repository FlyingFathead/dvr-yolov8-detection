# dvr-yolo8-detection

## Description

The `dvr-yolo8-detection` computer vision project is designed for real-time detection of humans, animals, or objects using the YOLOv8 model and OpenCV. The project supports real-time video streams via RTMP or USB webcams, includes CUDA GPU acceleration for enhanced performance, and provides options for saving detections and logging events.

## Features

- Real-time human / animal / object detection and alert system
- Runs on Python + YOLOv8 + OpenCV2
- Supports RTMP streams for real-time video sources
   - (a loopback example + nginx config example included)
- Supports USB webcams for real-time video sources
- Supports CUDA GPU acceleration, runs in CPU-only mode as well
- Detections can be automatically saved to images with a detection log
- Separate tool included for offline video file detections

## Overview

This project uses Python with YOLOv8 and OpenCV2 to perform object detection on either real-time video streams or batches of video files. It processes each video frame by frame, detects humans (by default; other specified and YOLOv8 supported types/objects can be added in as needed). The program can log the detections into a separate log file and additionally, save the detection frames with detected objects/entities highlighted and can send out an audio alert on detections via `pyttsx3`.

See the `config.ini` for configuration options.

Using CUDA-enabled OpenCV is recommended for faster operation. **Note: CUDA-enabled OpenCV2 needs to be compiled manually and installed separately and the compiling process is highly dependent on your hardware setup. You can look up the "Troubleshooting" portion of this page for some generic advice and an example build script for OpenCV with CUDA.**

Real-time detection also supports additional CUDA features such as CUDA video denoising (note: this feature requires CUDA, and is often available only when compiled from source).

- Configurable features options via `config.ini` are i.e.:
   - Your USB webcam or RTMP video source
   - Confidence threshold for detections
   - Enable or disable rescaling of video frames
   - CUDA-based video denoising (experimental)
   - Log detection details into a separate log file
   - Save frames with detected objects as image files
   - Model variant selection (i.e. YOLOv8n, YOLOv8s, YOLOv8m, etc.)
   - Plenty of other options to choose from...

## Requirements

- Python 3.6+ with the following modules:
   - torch
   - ultralytics (YOLO)
   - OpenCV (CUDA-enabled recommended for faster processing)
      - (`opencv-python` if CUDA is not available)
   - numpy
   - pyttsx3 (for text-to-speech alerts)
   - ffmpeg (for handling RTMP streams)
   - pytz
   - ffmpeg-python (for additional processing and loopbacks)

See `requirements.txt` for needed Python modules.

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

   **(Note: the pre-packaged `pip` modules are without CUDA support by default)**

   - For a full version (with GUI support, **recommended**):
     ```bash
     pip install opencv-python
     ```
   - For a headless version (no GUI support, **not recommended unless using offline batch processing only**):
     ```bash
     pip install opencv-python-headless
     ```

   - **For CUDA-enabled OpenCV**, you need to build it from source. Follow the instructions on the [OpenCV documentation](https://docs.opencv.org/master/d6/d15/tutorial_building_tegra_cuda.html) or take a peek at my build script [here](https://github.com/FlyingFathead/dvr-yolov8-detection/blob/main/utils/install_and_compile_opencv_with_cuda.sh) for potential tips on how to compile OpenCV with CUDA on Linux.

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

## Troubleshooting

### CUDA not found
- Make sure that you have all the necessary modules installed with CUDA enabled. In many cases, enabling CUDA support might require you to compile OpenCV directly from source, which can be a daunting task especially for those with no prior experience. 

- I have included my own OpenCV compile script [here](https://github.com/FlyingFathead/dvr-yolov8-detection/blob/main/utils/install_and_compile_opencv_with_cuda.sh), however I **do not recommend you run it** without understanding what it does. I have only used it to compile OpenCV with CUDA on a very platform and GPU-specific Ubuntu Linux environment (compiled during the summer of 2024), so it's _intended as a cheat sheet only, not as an install script_, as it might already be outdated as you read this, due to OpenCV's repositories being constantly updated. _Take what you want out of the compile script and and use it at your OWN RISK -- you have been warned._

- There are multiple other guides online for compiling OpenCV with CUDA support, each with varying degrees of success.

- The main real-time detection program does have a check on startup that prints out whether or not CUDA has been found and which GPU is in use, it can be useful for checking out your CUDA support status.

- Good luck! _(you'll need it, as well as patience)_

### What if I don't have a CUDA-capable GPU?
- You can still run the program in CPU-only mode, although it will be _extremely_ sluggish in comparison to CUDA GPU acceleration -- whereas with CUDA-acceleration enabled it can run on a high FPS with large video streams (i.e. multiplexed A/V streams), in CPU-only mode, even on higher end CPU's, you might not get too many frames per second. 

Should the model be too heavy to run on your computer in CPU-only mode, you can try changing the model size to a smaller one in the `config.ini`, or, as an alternative: try adjusting some of the rescale and frame rate options.

It should be noted that rescaling adds an extra step to the input video processing: hence, if possible, the optimal approach would be to reduce the resolution and frame rate from your device's end (i.e. if you're using OBS Studio's Virtual Camera, adjust OBS's stream output settings) and/or changing the model size to a smaller one.

## TODO

- More error catching in edge cases
- Refactoring and added modularity
- Setup scripts
- Threshold for sending out audio & other alerts 
   - (i.e.: X [number of] detections with Y confidence within Z seconds)
- Hooks for i.e. sending detections remotely to web servers, bot API's etc.

## Changes

- v0.151 - fallbacks for directories
- v0.1501 - fallback to non-CUDA modes if CUDA not supported
- v0.15 - direct USB webcam support added!
   - you can edit the `config.ini` and set the input to webcam
   - or, run the main program with: `--use_webcam --webcam_index X` (where X is your webcam index)
   - added `utils/get_webcams.py` - a tool that you can run to check for available webcams and their index numbers
- v0.1402 - detection logging into a file added
- v0.1401 - configparser added; now configurable via `config.ini`
- v0.140 - more double-checking as to configuration options and their availability

## Licensing

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. For non-commercial use, you are free to share and adapt the material with appropriate credit. For commercial use, please contact us to obtain a separate commercial license.

- **Non-Commercial Use**: Follow the terms of the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Commercial Use**: Contact the author at flyingfathead <@> protonmail.com or visit [https://github.com/FlyingFathead](https://github.com/FlyingFathead) to discuss licensing options.

## Contributing

Contributions are always welcome! You can even leave your own development ideas either here on GitHub or directly via mail to flyingfathead <@> protonmail dot com.

## Credits
Developed by [FlyingFathead](https://github.com/FlyingFathead), with the usual digital ghost code from ChaosWhisperer.

---

If you like it, let me know. :-)
