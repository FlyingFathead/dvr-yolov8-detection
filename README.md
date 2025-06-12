# DVR-YOLOv8-Detection

## Description

`dvr-yolov8-detection` is designed for real-time detection of humans, animals, or objects using the YOLOv8 model and OpenCV. Contrary to its name, we're now supporting models up to [**YOLOv11**](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)!

The program supports real-time video streams via RTMP or USB webcams, includes CUDA GPU acceleration for enhanced performance, and provides options for saving detections, triggering alerts and logging events.

The video preview can be run both in a GUI window and headless on a local web server using the included Flask web server setup.

## Features

- **Real-time human/animal/object detection and alert system**
- **(New!)** Now runs on the newest [YOLOv11](https://github.com/ultralytics/ultralytics?tab=readme-ov-file) model by default
- 🐳 A Dockerfile for Dockerized installs is also included.
- Runs on **Python + YOLOv8-11 + OpenCV2**
- Both GUI and headless web server versions (`Flask`), 2-in-1
  - Web server has i.e. easy-to-use image carousel for captured still viewing etc
- **Supports CUDA GPU acceleration**, CPU-only mode is also supported
- **RTMP streams** or **USB webcams** can be used for real-time video sources
  - _Includes a loopback example and NGINX configuration example for RTMP use (i.e. OBS Studio)_
- Set up separate minimum confidence zones with the included masking tool
- Name your regions and get alerted with zone names (i.e. on Telegram)
- Additional real-time alerts on detections are also supported via [**Telegram**](https://telegram.org)
- Detections can be automatically saved as images with a detection log
- Send detection data to any remote SSH/SFTP location
- Separate tool included for **offline video file detections** for DVR-type faster-than-realtime post-processing (see: `utils/`)

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

🐳 For Docker setup, see the **[DOCKER_SETUP.md](./DOCKER_SETUP.md)** for a guide.

---

# Setup

## Requirements

- **Python 3.6+** (Python 3.10.x recommended)
  - **Python modules:**
  - See [requirements.txt](./requirements.txt)
- **FFmpeg** 
- **Python 3.10.x**
- If you wish to use CUDA GPU acceleration, you will need:
  - A Nvidia GPU that supports CUDA
  - Install **CUDA 11.8 or higher** to enable GPU-accelerated processing
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

**(New in v0.155)**: The real-time detection now comes with a mini web server running on Flask that enables you to run the detection framework in your browser by default when `headless` and `enable_webserver` are set to `true`. This will make headless deployment very easy. Note that it listens on `0.0.0.0:5000` by default, which you might want to change (i.e. to `127.0.0.1:5000`) for safety reasons.

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

### Configuration

#### Running Headless / As A Web Server

Due to Docker being a popular install option, you can run the program headless and with a `Flask` based mini-web server included. Compared to the regular GUI verison, there is likely a small framerate dip and latency in the output, but other than that, the functionality is not too far off from the GUI variant. 

In headless Docker installs, make sure that `headless` and `enable_webserver` are both set to `true`.

#### Editing Program Configuration

You can configure the program's parameters by editing the `config.ini` file. This allows you to set program parameters such as the input source, input stream address, output directory, confidence threshold, model variant, stream URL, and more.

### (For RTMP Sources) Example NGINX Configuration

An example NGINX configuration is provided in `example-nginx.conf`. This config sets up an RTMP server that listens on `127.0.0.1:1935` and allows local clients to publish and play streams.

### (For RTMP Sources) RTMP Loopback Script

To stream and process the video in real-time, use the `ffmpeg_video_stream_loopback.sh` script. Ensure your streaming client (e.g., OBS Studio) is set to stream to `rtmp://127.0.0.1:1935/live`.

### (For RTMP Sources) Windows Users / Platform-Agnostic Loopback for RTMP

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

## Setup separate mask areas/zones (minimum confidence)

**Mask detection areas**: this is highly useful where detections need to be above certain threshold to be saved and registered with separate alerts. You can use the method to i.e. increase thresholds on the input image's detection areas to avoid false positives.

The masking can be done with a GUI rectangle painter util under `./utils/region_masker.py`, i.e.:

  ```bash
   python ./utils/region_masker.py
  ```

This will run a region masking utility that enables you to set special zones with a GUI interface (no headless mode yet!) and have it saved into a file (`./data/ignore_zones.json` by default; see `config.ini` on how to utilize the feature)

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
- **0.1624**
  - Additional relative path fallback fix in aggregation
  - Paths for webUI are now correctly constructed
- **0.1623** (Jun 12 2025)
  - Fixed fallback logic for directory saving; now all file saves are verified; two-level fallback for each file save
  - Web server tries to fetch the file from both the intended and the fallback location when needed
- **0.1622**
  - **Feature: Implemented Lazy Loading for Web UI Detection Images.**
    - Aggregated detection entries are now assigned unique UUIDs during processing and persistence.
    - The main web UI (`/`) now sends only detection summaries initially, significantly reducing the initial page load size and improving performance, especially on slow connections.
    - The list of image filenames (`full_frame`, `detection_area`) for a specific aggregated detection is now fetched *on-demand* when the user clicks its "View Images" button.
    - Added a new API endpoint `/api/detection_images/<uuid>` to serve image lists for individual detections.
    - Updated frontend JavaScript to handle fetching image lists via the new API before displaying the image carousel.
  - **Fix:** Image carousel now correctly defaults to showing the `detection_area` image first (if available), only showing `full_frame` initially if `detection_area` is missing or upon user clicking the "Swap Type" button.
  - **Fix:** Adjusted CSS (`max-width`, `max-height`, `object-fit`) for the modal image (`#modal-image`) to ensure both `detection_area` and larger `full_frame` images are properly constrained within the viewport without overflow.
  - **Util:** Added `utils/uuid_inserter.py` script. This utility can process an existing `aggregated_detections.json` file (specified via command line or loaded from `config.ini`), creates a backup, and adds missing UUIDs to older entries, allowing them to be compatible with the new lazy-load
- **v0.1621**
  - **Web server updates:**
  - Switched to use `waitress` (`pip install -U waitress` or use the `requirements.txt`)
  - `waitress` = better threading and multitasking with `Flask`
  - MJPEG preview quality in webUI previews now configurable from `config.ini`
    => under `[webserver]` => `mjpeg_quality`
  - webui improvements: detections are now categorized under date headers for improved readability
- **v0.1620**
  - Disabled telemetry in Ultralytics modules by default
  - Added a printout on startup to display Ultralytics settings
- **v0.1619**
  - **Stream preview in the WebUI can now be swapped between HLS and MJPEG**
  - (HLS is sourced from the source RTMP stream; higher frame rates without recoding)
  - See `config.ini` => `preview_method = mjpeg` (swap to `hls` for HLS)
  - More options under `[hls]` section in the `config.ini`
- **v0.1618**
  - Named region masking is here, with critical thresholds
  - Use the renewd `./utils/region_masker.py` to set up your region names and their critical thresholds
  - Currently, Telegram alerts support notifications that emphasize the alert on critical thresholds
- **v0.1617**
  - Web server: added better navigation detection image browsing (first/last, skip 10 forward/back)
  - viewport fixes for image carousel
- **v0.1616**
  - **Now using YOLOv11 by default** 
  - YOLOv11 is (as of writing this) the latest YOLO version from [ultralytics](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)
  - Update your `ultralytics` packages with i.e. `pip install -U ultralytics`
  - Switch to any version you like from `config.ini`
- **v0.1615**
  - **New feature: Mask detection areas** -- this is highly useful where detections need to be above certain threshold to be saved and registered with separate alerts. You can use the method to i.e. increase thresholds on the input image's detection areas to avoid false positives.
  - The masking can be done with a GUI rectangle painter util under `./utils/region_masker.py`, i.e. like so:
  ```bash
   python ./utils/region_masker.py
  ```
  - This will run a region masking utility that enables you to set special zones with a GUI interface (no headless mode yet!) and have it saved into a file (`./data/ignore_zones.json` by default; see `config.ini` for more configuration options)
  - fixed poller startup message in `utils/detection_audio_poller.py`
- **v0.1614.3**
  - preferred CUDA device can now be selected under `[hardware]` from `config.ini`
- **v0.1614**
  - More reactive detection aggregation updates to webui
- **v0.1613**
  - Even more load balancing fixes; aggregation parsing improvements
- **v0.1612**
  - Improved thread and queue handling (non-blocking remote sync and other actions, etc)
  - Maximum queue sizes can be set separately for saved frames and remote syncs (see `config.ini`)
- **v0.1611**
  - TTS handling changes; test message on startup
  - (TODO) Firejail users may still encounter issues due to audio routing inside Firejail instances
- **v0.1610**
  - Remote sync features added, bugfixes
  - Firejail & venv switches when remote syncing via SSH/SCP is enabled
- **v0.1609**
  - Remote sync detection logs & frames to a remote SFTP/SSH server with either system `scp` or `paramiko`
  - Can be configured and enabled/disabled in `config.ini` under the `remote_sync` options
- **v0.1608**
  - Added persistence to aggregated detections (esp. for web server use)
  - Can be enabled or disabled in `config.ini` with the following parameters:
  - `enable_persistent_aggregated_detections = true`
  - `aggregated_detections_file = ./logs/aggregated_detections.json`
  - Program version display added to universal `version.py` file  
- **v0.1607** 
  - **New: Get detection alerts via [Telegram](https://core.telegram.org/api)** (optional)
  - Use [@BotFather](https://t.me/BotFather) on Telegram to create a bot token
  - Set your userid(s) (can be multiple users, comma separated) and the bot API token as environment variables:
    - `DVR_YOLOV8_ALLOWED_TELEGRAM_USERS`
      - allowed users/send alerts to (comma-separated list)
    - `DVR_YOLOV8_TELEGRAM_BOT_TOKEN`
      - your Telegram bot API token for alerts
- **v0.1606** 
  - Performance improvements: 
    - switched to PyAV-based handling for better RTMP stream reliability with less CPU load
  - UI/UX improvements:
    - Web server UI/UX enhancements in image carousel browsing
    - shows detection area by default, swappable between full frame/area
    - clicking on the image now shows its original version
    - better error/exception catching overall
    - better webUI scaling in various devices etc
- **v0.1605** Overall compatibility & bug fixes
  - Detection image carousel beta over webUI
  - If detection saving is enabled, images can be viewed from webUI
- **v0.1604** Frame queue sizes now configurable
  - Helps in I/O performance issues when saving detections
  - `config.ini` => `[performance]` => `frame_queue_size`
- **v0.1603** New configuration for saving
  - Choose to save the detection area, the whole frame, or both.
  - _(see `config.ini` => `save_full_frames` and `save_detection_areas`)_
- **v0.1602** Queuing on image saving
  - Should reduce lag on most systems, even with larger frames
- **v0.1601** Active access logging for webUI connections; improved
  - Access via webUI is logged by default to `logs/access.log`
  - See `config.ini` for more options
- **v0.160** (Oct-13-2024) WebUI access logging added
  - Can be enabled/disabled and defined in `config.ini`
- **v0.159** (Oct-12-2024) Fixes to the detection saving logic
- **v0.158** (Oct-11-2024) **Even more webUI updates**
  - Human detections get aggregated in the webUI within a cooldown period
  - (default cooldown period: 30 seconds)
- **v0.157** (Oct-11-2024) **webUI updates**
  - Better refreshed data via AJAX
  - Minimized lock holding time `web_server.py` => better FPS
  - `webserver_max_fps` value to cap the framerate on the webUI for better performance
- **v0.156** (Oct-11-2024) **Detection graphs in web UI**
  - Added `matplotlib` based detection graphs to the web UI
  - (selectable between 1hr/24hrs/week/month/year)
- **v0.155** (Oct-11-2024) **Now comes with a Flask web server!**
  - The video feed can be monitored real-time using the web interface
  - Added a `Flask` mini web server to take care of the streams
  - `enable_webserver` and `headless` both set to `true` by default
  - Server listens at `0.0.0.0:5000` (see `config.ini` for more)  
  - This enables quick deployment especially in headless / Docker setups
- **v0.154** (Oct-10-2024) 🐳 **Dockerized Setup Now Available!** 🐳
  - Headless mode added for non-GUI/Docker/detection-only modes
    - enable in `config.ini` with `headless = true`
    - or, run the program with `--headless` flag
  - Added Docker as an install method to ease the setup process
  - additional installation guides
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