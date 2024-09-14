#!/usr/bin/env python3
#
# yolov8_live_rtmp_stream_detection.py
# Sept 14 // 2024
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/dvr-yolov8-detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Version number
version_number = 0.153

import cv2
import torch
import logging
import numpy as np

# time and tz related
import time
from datetime import datetime
import pytz

import pyttsx3
import os
from ultralytics import YOLO
import threading
from threading import Thread, Event, Lock
from queue import Queue
import configparser
import argparse
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from `config.ini` file with case-sensitive keys
def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    config.optionxform = str  # Override optionxform to make keys case-sensitive
    config.read(config_path)
    return config

# Load configuration
config = load_config()

# Assign configurations to variables
DEFAULT_CONF_THRESHOLD = config.getfloat('general', 'default_conf_threshold')
DEFAULT_MODEL_VARIANT = config.get('general', 'default_model_variant')
USE_WEBCAM = config.getboolean('input', 'use_webcam', fallback=False)
WEBCAM_INDEX = config.getint('input', 'webcam_index', fallback=0)
STREAM_URL = config.get('stream', 'stream_url')
DRAW_RECTANGLES = config.getboolean('detection', 'draw_rectangles')
SAVE_DETECTIONS = config.getboolean('detection', 'save_detections')
IMAGE_FORMAT = config.get('detection', 'image_format')
USE_ENV_SAVE_DIR = config.getboolean('detection', 'use_env_save_dir')
ENV_SAVE_DIR_VAR = config.get('detection', 'env_save_dir_var')
DEFAULT_SAVE_DIR = config.get('detection', 'default_save_dir')
FALLBACK_SAVE_DIR = config.get('detection', 'fallback_save_dir')
RETRY_DELAY = config.getint('detection', 'retry_delay')
MAX_RETRIES = config.getint('detection', 'max_retries')
RESCALE_INPUT = config.getboolean('detection', 'rescale_input')
TARGET_HEIGHT = config.getint('detection', 'target_height')
DENOISE = config.getboolean('detection', 'denoise')
USE_PROCESS_FPS = config.getboolean('detection', 'use_process_fps')
PROCESS_FPS = config.getint('detection', 'process_fps')
TIMEOUT = config.getint('detection', 'timeout')
TTS_COOLDOWN = config.getint('detection', 'tts_cooldown')
# Logging options
ENABLE_DETECTION_LOGGING_TO_FILE = config.getboolean('logging', 'enable_detection_logging_to_file')
LOG_DIRECTORY = config.get('logging', 'log_directory')
LOG_FILE = config.get('logging', 'log_file')
DETECTION_LOG_FILE = config.get('logging', 'detection_log_file')

# New configuration for creating date-based subdirectories
CREATE_DATE_SUBDIRS = config.getboolean('detection', 'create_date_subdirs', fallback=False)

# Define the main logger
main_logger = logging.getLogger('main')
main_logger.setLevel(logging.INFO)

# Clear existing handlers if they exist
if main_logger.hasHandlers():
    main_logger.handlers.clear()

# Configure main logger for console output
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
main_logger.addHandler(stream_handler)

# Configure detection logger if enabled
detection_logger = logging.getLogger('detection')
detection_logger.setLevel(logging.INFO)

# Check if detection logging to file is enabled
if ENABLE_DETECTION_LOGGING_TO_FILE:
    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)
    detection_log_path = os.path.join(LOG_DIRECTORY, DETECTION_LOG_FILE)
    detection_file_handler = logging.FileHandler(detection_log_path)
    detection_file_handler.setLevel(logging.INFO)
    detection_file_handler.setFormatter(formatter)
    detection_logger.addHandler(detection_file_handler)
    main_logger.info(f"Detection logging to file is enabled. Logging to: {detection_log_path}")
else:
    main_logger.info("Detection logging to file is disabled.")

# Timekeeping and frame counting
last_log_time = time.time()

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_lock = Lock()
last_tts_time = 0
tts_thread = None
tts_stop_event = Event()

# Load the YOLOv8 model
def load_model(model_variant=DEFAULT_MODEL_VARIANT):
    try:
        model = YOLO(model_variant)
        # Check if CUDA is available
        if torch.cuda.is_available():
            model.to('cuda')
            main_logger.info("Using CUDA for model inference.")
        else:
            model.to('cpu')
            main_logger.warning("CUDA not available, using CPU for model inference.")
        return model
    except Exception as e:
        logging.error(f"Error loading model {model_variant}: {e}")
        raise

# Initialize model
model = load_model(DEFAULT_MODEL_VARIANT)

# Get available CUDA GPU and its details
def log_cuda_info():
    if not torch.cuda.is_available():
        logging.warning("No CUDA devices detected. Running on CPU.")
        return

    num_gpus = torch.cuda.device_count()
    logging.info(f"Number of CUDA devices detected: {num_gpus}")

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_capability = torch.cuda.get_device_capability(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # Convert bytes to GB
        logging.info(f"CUDA Device {i}: {gpu_name}")
        logging.info(f"  Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
        logging.info(f"  Total Memory: {gpu_memory:.2f} GB")

    # Log the current device being used
    current_device = torch.cuda.current_device()
    current_gpu_name = torch.cuda.get_device_name(current_device)
    logging.info(f"Using CUDA Device {current_device}: {current_gpu_name}")

# Function to get the base save directory (without date subdirs)
def get_base_save_dir():
    base_save_dir = None

    # 1. Use environment-specified directory
    if USE_ENV_SAVE_DIR:
        env_dir = os.getenv(ENV_SAVE_DIR_VAR)
        if env_dir and os.path.exists(env_dir) and os.access(env_dir, os.W_OK):
            main_logger.info(f"Using environment-specified save directory: {env_dir}")
            base_save_dir = env_dir
        else:
            main_logger.warning(f"Environment variable {ENV_SAVE_DIR_VAR} is set but the directory does not exist or is not writable. Checked path: {env_dir}")

    # 2. Fallback to fallback_save_dir
    if not base_save_dir and FALLBACK_SAVE_DIR:
        if os.path.exists(FALLBACK_SAVE_DIR) and os.access(FALLBACK_SAVE_DIR, os.W_OK):
            main_logger.info(f"Using fallback save directory: {FALLBACK_SAVE_DIR}")
            base_save_dir = FALLBACK_SAVE_DIR
        else:
            main_logger.warning(f"Fallback save directory {FALLBACK_SAVE_DIR} does not exist or is not writable. Attempting to create it.")
            try:
                os.makedirs(FALLBACK_SAVE_DIR, exist_ok=True)
                if os.access(FALLBACK_SAVE_DIR, os.W_OK):
                    main_logger.info(f"Created and using fallback save directory: {FALLBACK_SAVE_DIR}")
                    base_save_dir = FALLBACK_SAVE_DIR
                else:
                    main_logger.warning(f"Fallback save directory {FALLBACK_SAVE_DIR} is not writable after creation.")
            except Exception as e:
                main_logger.error(f"Failed to create fallback save directory: {FALLBACK_SAVE_DIR}. Error: {e}")

    # 3. Use default_save_dir
    if not base_save_dir and DEFAULT_SAVE_DIR:
        if os.path.exists(DEFAULT_SAVE_DIR) and os.access(DEFAULT_SAVE_DIR, os.W_OK):
            main_logger.info(f"Using default save directory: {DEFAULT_SAVE_DIR}")
            base_save_dir = DEFAULT_SAVE_DIR
        else:
            main_logger.warning(f"Default save directory {DEFAULT_SAVE_DIR} does not exist or is not writable. Attempting to create it.")
            try:
                os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
                if os.access(DEFAULT_SAVE_DIR, os.W_OK):
                    main_logger.info(f"Created and using default save directory: {DEFAULT_SAVE_DIR}")
                    base_save_dir = DEFAULT_SAVE_DIR
                else:
                    main_logger.warning(f"Default save directory {DEFAULT_SAVE_DIR} is not writable after creation.")
            except Exception as e:
                main_logger.error(f"Failed to create default save directory: {DEFAULT_SAVE_DIR}. Error: {e}")

    # 4. Final fallback to a hardcoded directory (optional)
    if not base_save_dir:
        final_fallback_dir = os.path.join(os.path.dirname(__file__), 'final_fallback_detections')
        if os.path.exists(final_fallback_dir) and os.access(final_fallback_dir, os.W_OK):
            main_logger.info(f"Using final hardcoded fallback save directory: {final_fallback_dir}")
            base_save_dir = final_fallback_dir
        else:
            try:
                os.makedirs(final_fallback_dir, exist_ok=True)
                if os.access(final_fallback_dir, os.W_OK):
                    main_logger.info(f"Created and using final hardcoded fallback save directory: {final_fallback_dir}")
                    base_save_dir = final_fallback_dir
                else:
                    main_logger.warning(f"Final fallback save directory {final_fallback_dir} is not writable after creation.")
            except Exception as e:
                main_logger.error(f"Failed to create final hardcoded fallback save directory: {final_fallback_dir}. Error: {e}")

    # Raise error if no writable directory is found
    if not base_save_dir:
        raise RuntimeError("No writable save directory available.")

    return base_save_dir

# Initialize SAVE_DIR and CURRENT_DATE
SAVE_DIR_BASE = get_base_save_dir()
CURRENT_DATE = datetime.now().date()

# Function to get the current save directory with date-based subdirectories
def get_current_save_dir():
    global CURRENT_DATE, SAVE_DIR_BASE

    if CREATE_DATE_SUBDIRS:
        new_date = datetime.now().date()
        if new_date != CURRENT_DATE:
            CURRENT_DATE = new_date
            # Update SAVE_DIR with new date-based subdirectories
            year = CURRENT_DATE.strftime("%Y")
            month = CURRENT_DATE.strftime("%m")
            day = CURRENT_DATE.strftime("%d")
            date_subdir = os.path.join(SAVE_DIR_BASE, year, month, day)
            try:
                os.makedirs(date_subdir, exist_ok=True)
                main_logger.info(f"Date changed. Created/navigated to new date-based subdirectory: {date_subdir}")
                return date_subdir
            except Exception as e:
                main_logger.error(f"Failed to create new date-based subdirectory {date_subdir}: {e}")
                # Fallback to base_save_dir if subdirectory creation fails
                return SAVE_DIR_BASE
        else:
            # Ensure the date-based subdirectory exists
            year = CURRENT_DATE.strftime("%Y")
            month = CURRENT_DATE.strftime("%m")
            day = CURRENT_DATE.strftime("%d")
            date_subdir = os.path.join(SAVE_DIR_BASE, year, month, day)
            if not os.path.exists(date_subdir):
                try:
                    os.makedirs(date_subdir, exist_ok=True)
                    main_logger.info(f"Created date-based subdirectory: {date_subdir}")
                except Exception as e:
                    main_logger.error(f"Failed to create date-based subdirectory {date_subdir}: {e}")
            return date_subdir
    else:
        return SAVE_DIR_BASE

# Initialize the initial SAVE_DIR
SAVE_DIR = get_current_save_dir()

# Function to log detection details
def log_detection_details(detections, frame_count, timestamp):
    if ENABLE_DETECTION_LOGGING_TO_FILE:
        for detection in detections:
            x1, y1, x2, y2, confidence, class_idx = detection
            detection_logger.info(f"Detection: Frame {frame_count}, Timestamp {timestamp}, Coordinates: ({x1}, {y1}), ({x2}, {y2}), Confidence: {confidence:.2f}")

# Function to resize while maintaining aspect ratio
def resize_frame(frame, target_height):
    """Resize the frame to the target height while maintaining the aspect ratio."""
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    resized_frame = cv2.resize(frame, (new_width, target_height))
    return resized_frame

# Function to announce detection using a separate thread
def announce_detection():
    global last_tts_time
    while not tts_stop_event.is_set():
        current_time = time.time()
        if current_time - last_tts_time >= TTS_COOLDOWN:
            with tts_lock:
                tts_engine.say("Human detected!")
                tts_engine.runAndWait()
            last_tts_time = current_time
        time.sleep(1)

# Function to save detection image with date check
def save_detection_image(frame, detection_count):
    global SAVE_DIR
    try:
        # Update SAVE_DIR based on current date
        SAVE_DIR = get_current_save_dir()
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Format timestamp as YYYYMMDD-HHMMSS
        filename = os.path.join(SAVE_DIR, f"{timestamp}_{detection_count}.{IMAGE_FORMAT}")  # Ensure SAVE_DIR already includes date subdirs
        cv2.imwrite(filename, frame)
        main_logger.info(f"Saved detection image: {filename}")
    except Exception as e:
        main_logger.error(f"Error saving detection image: {e}")

# Check if the CUDA denoising function is available
def cuda_denoising_available():
    try:
        _ = cv2.cuda.createFastNlMeansDenoisingColored
        main_logger.info("CUDA denoising is available.")
        return True
    except AttributeError:
        main_logger.warning("CUDA denoising is not available.")
        return False

# Frame capture thread
def frame_capture_thread(stream_url, use_webcam, webcam_index, frame_queue, stop_event):
    retries = 0
    cap = None

    if use_webcam:
        main_logger.info(f"Using webcam with index {webcam_index}")
        cap = cv2.VideoCapture(webcam_index)
    else:
        while retries < MAX_RETRIES:
            main_logger.info(f"Attempting to open video stream: {stream_url}")

            # Set FFmpeg options
            ffmpeg_options = {
                'rtmp_buffer': '1000',  # Increase buffer size
                'max_delay': '5000000',  # Increase maximum delay
                'stimeout': '5000000'  # Socket timeout
            }
            options = ' '.join([f'-{k} {v}' for k, v in ffmpeg_options.items()])
            cap = cv2.VideoCapture(f'{stream_url} {options}')

            # Set the timeout for the video stream
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT * 1000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT * 1000)

            if cap.isOpened():
                main_logger.info("Successfully opened video stream.")
                break
            else:
                main_logger.error(f"Unable to open video stream: {stream_url}. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                retries += 1

    if cap is None or not cap.isOpened():
        main_logger.error(f"Failed to open video source after {MAX_RETRIES} retries.")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            main_logger.warning("Failed to read frame from source.")
            time.sleep(1)
            continue
        frame_queue.put(frame)

    cap.release()
    main_logger.info("Video source closed.")

# Frame processing thread
def frame_processing_thread(frame_queue, stop_event, conf_threshold, draw_rectangles, denoise, process_fps, use_process_fps, detection_ongoing):
    global last_tts_time, tts_thread, tts_stop_event
    model = load_model(DEFAULT_MODEL_VARIANT)
    use_cuda_denoising = cuda_denoising_available()
    detection_count = 0
    last_log_time = time.time()
    total_frames = 0
    detecting_human = False

    # Create a named window with the ability to resize
    cv2.namedWindow('Real-time Human Detection', cv2.WINDOW_NORMAL)

    while not stop_event.is_set() or not frame_queue.empty():
        if not frame_queue.empty():
            frame = frame_queue.get()
            start_time = time.time()

            if RESCALE_INPUT:
                resized_frame = resize_frame(frame, TARGET_HEIGHT)
            else:
                resized_frame = frame

            if denoise:
                try:
                    if use_cuda_denoising:
                        gpu_frame = cv2.cuda_GpuMat()
                        gpu_frame.upload(resized_frame)
                        denoised_gpu = cv2.cuda.createFastNlMeansDenoisingColored(gpu_frame, None, 3, 3, 7)
                        denoised_frame = denoised_gpu.download()
                    else:
                        denoised_frame = cv2.fastNlMeansDenoisingColored(resized_frame, None, 3, 3, 7, 21)
                except cv2.error as e:
                    main_logger.error(f"Error applying denoising: {e}")
                    denoised_frame = resized_frame
            else:
                denoised_frame = resized_frame

            results = model.predict(source=denoised_frame, conf=conf_threshold, classes=[0], verbose=False)
            detections = results[0].boxes.data.cpu().numpy()

            if detections.size > 0:
                if not detecting_human:
                    detecting_human = True
                    detection_ongoing.set()
                    detection_count += 1
                    main_logger.info(f"Detections found: {detections}")
                for detection in detections:
                    x1, y1, x2, y2, confidence, class_idx = detection
                    if draw_rectangles:
                        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(denoised_frame.shape[1], int(x2)), min(denoised_frame.shape[0], int(y2))
                        cv2.rectangle(denoised_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'Person: {confidence:.2f}'
                        cv2.putText(denoised_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        main_logger.info(f"Rectangle drawn: {x1, y1, x2, y2} with confidence {confidence:.2f}")
                if SAVE_DETECTIONS:
                    save_detection_image(denoised_frame, detection_count)

                # Calculate the current timestamp using the local system's timezone
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')

                # Log the detection details
                log_detection_details(detections, total_frames, timestamp)

                # Start the TTS announcement in a separate thread if not already running
                if tts_thread is None or not tts_thread.is_alive():
                    tts_stop_event.clear()
                    tts_thread = threading.Thread(target=announce_detection)
                    tts_thread.start()

            else:
                if detecting_human:
                    detecting_human = False
                    detection_ongoing.clear()
                    tts_stop_event.set()  # Stop TTS when no human is detected

            # Display the denoised frame
            cv2.imshow('Real-time Human Detection', denoised_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            total_frames += 1
            if time.time() - last_log_time >= 1:
                fps = total_frames / (time.time() - last_log_time)
                main_logger.info(f'Processed {total_frames} frames at {fps:.2f} FPS')
                last_log_time = time.time()
                total_frames = 0

            if use_process_fps:
                frame_time = time.time() - start_time
                frame_interval = 1.0 / process_fps
                time_to_wait = frame_interval - frame_time
                if time_to_wait > 0:
                    time.sleep(time_to_wait)

    tts_stop_event.set()  # Ensure TTS stops if the program is stopping
    cv2.destroyAllWindows()

# When the user wants to exit
def signal_handler(sig, frame):
    main_logger.info("Interrupt received, stopping, please wait for the program to finish...")
    stop_event.set()

# Main
if __name__ == "__main__":
    log_cuda_info()  # Add this line
    parser = argparse.ArgumentParser(description="YOLOv8 RTMP Stream Human Detection")
    parser.add_argument("--stream_url", type=str, default=STREAM_URL, help="URL of the RTMP stream")
    parser.add_argument("--use_webcam", type=bool, default=USE_WEBCAM, help="Use webcam for video input")
    parser.add_argument("--webcam_index", type=int, default=WEBCAM_INDEX, help="Index number of the webcam to use")
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold for detections")
    parser.add_argument("--model_variant", type=str, default=DEFAULT_MODEL_VARIANT, help="YOLOv8 model variant to use")
    parser.add_argument("--draw_rectangles", type=bool, default=DRAW_RECTANGLES, help="Draw rectangles around detected objects")
    parser.add_argument("--save_detections", type=bool, default=SAVE_DETECTIONS, help="Save images with detections")
    parser.add_argument("--image_format", type=str, default=IMAGE_FORMAT, help="Image format for saving detections (jpg or png)")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save detection images")  # Changed default to None
    # Removed fallback_save_dir from arguments as it's handled via config.ini
    parser.add_argument("--retry_delay", type=int, default=RETRY_DELAY, help="Delay between retries to connect to the stream")
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES, help="Maximum number of retries to connect to the stream")
    parser.add_argument("--rescale_input", type=bool, default=RESCALE_INPUT, help="Rescale the input frames")
    parser.add_argument("--target_height", type=int, default=TARGET_HEIGHT, help="Target height for rescaling the frames")
    parser.add_argument("--denoise", type=bool, default=DENOISE, help="Toggle denoising on/off")
    parser.add_argument("--process_fps", type=int, default=PROCESS_FPS, help="Frames per second for processing")
    parser.add_argument("--use_process_fps", type=bool, default=USE_PROCESS_FPS, help="Whether to use custom processing FPS")
    parser.add_argument("--timeout", type=int, default=TIMEOUT, help="Timeout for the video stream in seconds")

    args = parser.parse_args()

    # Initialize SAVE_DIR using the updated get_save_dir function
    try:
        SAVE_DIR = get_current_save_dir()
        main_logger.info(f"Saving detections to directory: {SAVE_DIR}")
    except RuntimeError as e:
        main_logger.error(f"Error initializing save directory: {e}")
        sys.exit(1)

    # Update other configurations based on arguments
    STREAM_URL = args.stream_url
    USE_WEBCAM = args.use_webcam
    WEBCAM_INDEX = args.webcam_index    
    DRAW_RECTANGLES = args.draw_rectangles
    SAVE_DETECTIONS = args.save_detections
    IMAGE_FORMAT = args.image_format
    RETRY_DELAY = args.retry_delay
    MAX_RETRIES = args.max_retries
    RESCALE_INPUT = args.rescale_input
    TARGET_HEIGHT = args.target_height
    DENOISE = args.denoise
    PROCESS_FPS = args.process_fps
    USE_PROCESS_FPS = args.use_process_fps
    TIMEOUT = args.timeout

    frame_queue = Queue(maxsize=10)
    stop_event = Event()
    detection_ongoing = Event()

    # Register signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize threads
    capture_thread = Thread(target=frame_capture_thread, args=(STREAM_URL, USE_WEBCAM, WEBCAM_INDEX, frame_queue, stop_event))
    processing_thread = Thread(target=frame_processing_thread, args=(frame_queue, stop_event, args.conf_threshold, DRAW_RECTANGLES, DENOISE, PROCESS_FPS, USE_PROCESS_FPS, detection_ongoing))

    try:
        capture_thread.start()
        processing_thread.start()

        capture_thread.join()
        processing_thread.join()
    
    finally:
        # Ensure TTS thread is stopped
        tts_stop_event.set()
        if tts_thread is not None:
            tts_thread.join()
        main_logger.info("Cleaning up threads and resources.")
        cv2.destroyAllWindows()
        main_logger.info("Program exited cleanly.")
