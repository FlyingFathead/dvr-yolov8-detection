# yolov8_live_rtmp_stream_detection.py
# (Updated Oct 13, 2024)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/dvr-yolov8-detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Version number
version_number = 0.161

import cv2
import torch
import logging
import numpy as np
from utils import hz_line

# Time and timezone related
import time
from datetime import datetime
import pytz

from collections import deque
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

# Import web server functions
from web_server import start_web_server, set_output_frame

hz_line()
print(f"::: dvr-yolov8-detection v{version_number} | https://github.com/FlyingFathead/dvr-yolov8-detection/")
hz_line()

# Shared data structures
detections_list = deque(maxlen=100)  # Store up to 100 latest detections on web UI
logs_list = deque(maxlen=100)        # Store up to 100 latest logs on web UI
detection_timestamps = deque(maxlen=10000)  # Store up to 10,000 timestamps

# Locks for thread safety
timestamps_lock = threading.Lock()
detections_lock = threading.Lock()
logs_lock = threading.Lock()

# Load configuration from `config.ini` file with case-sensitive keys
def load_config(config_path=None):
    try:
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"Script directory resolved to: {script_dir}", flush=True)
            config_path = os.path.join(script_dir, 'config.ini')
        
        print(f"Attempting to load config from: {config_path}", flush=True)
        config = configparser.ConfigParser()
        # config.optionxform = str  # Make keys case-sensitive
        read_files = config.read(config_path)
        
        if not read_files:
            print(f"Warning: Config file {config_path} not found or is empty. Using default settings.", flush=True)
        else:
            print(f"Config file {config_path} loaded successfully.", flush=True)
        
        return config
    except Exception as e:
        print(f"Error loading config: {e}", flush=True)
        raise

# Load configuration
print("Starting configuration loading...", flush=True)
config = load_config()
print("Configuration loading completed.", flush=True)

# Assign configurations to variables
HEADLESS = config.getboolean('detection', 'headless', fallback=False)
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
# Web server configuration
ENABLE_WEBSERVER = config.getboolean('webserver', 'enable_webserver', fallback=False)
WEBSERVER_HOST = config.get('webserver', 'webserver_host', fallback='0.0.0.0')
WEBSERVER_PORT = config.getint('webserver', 'webserver_port', fallback=5000)

# List handler for webUI logging
class ListHandler(logging.Handler):
    def __init__(self, logs_list, logs_lock):
        super().__init__()
        self.logs_list = logs_list
        self.logs_lock = logs_lock

    def emit(self, record):
        log_entry = self.format(record)
        with self.logs_lock:
            self.logs_list.append(log_entry)

# Centralized Logging Configuration
def setup_logging():
    # Create a root logger with StreamHandler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Prevent root logger from propagating to higher loggers (if any)
    root_logger.propagate = False
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    # Create and add StreamHandler to root logger
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    
    # Define the main logger
    main_logger = logging.getLogger('main')
    main_logger.setLevel(logging.INFO)
    main_logger.propagate = False  # Prevent propagation to root
    
    # Add ListHandler to main_logger to capture WARNING and ERROR logs
    list_handler = ListHandler(logs_list, logs_lock)
    list_handler.setLevel(logging.WARNING)  # Capture WARNING and ERROR logs
    list_handler.setFormatter(formatter)
    main_logger.addHandler(list_handler)
    
    # **Add StreamHandler to main_logger for INFO level logs**
    stream_handler_main = logging.StreamHandler(sys.stdout)
    stream_handler_main.setLevel(logging.INFO)
    stream_handler_main.setFormatter(formatter)
    main_logger.addHandler(stream_handler_main)
    
    # Define detection logger
    detection_logger = logging.getLogger('detection')
    detection_logger.setLevel(logging.INFO)
    detection_logger.propagate = False  # Prevent propagation to root
    
    # Initialize detection_log_path
    detection_log_path = None
    
    # Add FileHandler to detection_logger if enabled
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
    
    # Define web_server logger
    web_server_logger = logging.getLogger('web_server')
    web_server_logger.setLevel(logging.INFO)
    web_server_logger.propagate = False  # Prevent propagation to root
    
    # Add StreamHandler to web_server_logger if not already present
    if not web_server_logger.hasHandlers():
        web_stream_handler = logging.StreamHandler(sys.stdout)
        web_stream_handler.setLevel(logging.INFO)
        web_stream_handler.setFormatter(formatter)
        web_server_logger.addHandler(web_stream_handler)
    
    return main_logger, detection_logger, web_server_logger, detection_log_path

# Initialize logging
main_logger, detection_logger, web_server_logger, detection_log_path = setup_logging()

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
    main_logger.info(f"Current SAVE_DIR is: {SAVE_DIR}")
    main_logger.info("Attempting to save detection image.")    
    try:
        # Update SAVE_DIR based on current date
        SAVE_DIR = get_current_save_dir()
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Format timestamp as YYYYMMDD-HHMMSS
        filename = os.path.join(SAVE_DIR, f"{timestamp}_{detection_count}.{IMAGE_FORMAT}")  # Ensure SAVE_DIR already includes date subdirs
        if not cv2.imwrite(filename, frame):
            main_logger.error(f"Failed to save detection image: {filename}")
        else:
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
def frame_processing_thread(frame_queue, stop_event, conf_threshold, draw_rectangles, denoise, process_fps, use_process_fps, detection_ongoing, headless):
    global last_tts_time, tts_thread, tts_stop_event
    model = load_model(DEFAULT_MODEL_VARIANT)
    use_cuda_denoising = cuda_denoising_available()
    detection_count = 0
    last_log_time = time.time()
    total_frames = 0
    detecting_human = False

    if not headless:
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
                main_logger.info(f"SAVE_DETECTIONS is set to: {SAVE_DETECTIONS}")
                if not detecting_human:
                    detecting_human = True
                    detection_ongoing.set()
                    detection_count += 1
                    main_logger.info(f"Detections found: {detections}")

                # Save image for every detection frame
                if SAVE_DETECTIONS:
                    main_logger.info("Saving detection image.")
                    try:
                        save_detection_image(denoised_frame, detection_count)
                    except Exception as e:
                        main_logger.error(f"Error during save_detection_image: {e}")

                # **Assign timestamp here**
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                for detection in detections:
                    x1, y1, x2, y2, confidence, class_idx = detection

                    # Convert NumPy data types to native Python types
                    detection_info = {
                        'frame_count': int(total_frames),
                        'timestamp': timestamp,
                        'coordinates': (
                            int(x1), int(y1), int(x2), int(y2)
                        ),
                        'confidence': float(confidence)
                    }                    

                    # detection_info = {
                    #     'frame_count': total_frames,
                    #     'timestamp': timestamp,
                    #     'coordinates': (x1, y1, x2, y2),
                    #     'confidence': confidence
                    # }

                    # for webui; record detection timestamp
                    current_time = time.time()
                    with timestamps_lock:
                        detection_timestamps.append(current_time)

                    # for webui; use appendleft to show recent detections first
                    with detections_lock:
                        detections_list.appendleft(detection_info)

                    if draw_rectangles:
                        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(denoised_frame.shape[1], int(x2)), min(denoised_frame.shape[0], int(y2))
                        if not headless:
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

            if ENABLE_WEBSERVER:
                # Send the processed frame to the web server
                set_output_frame(denoised_frame)
            elif not headless:
                # Display the denoised frame
                cv2.imshow('Real-time Human Detection', denoised_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

            # if not headless:
            #     # Display the denoised frame
            #     cv2.imshow('Real-time Human Detection', denoised_frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         stop_event.set()
            #         break

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
    if not headless:
        cv2.destroyAllWindows()

# When the user wants to exit
HEADLESS = False  # Initialize globally

def signal_handler(sig, frame):
    main_logger.info("Interrupt received, stopping, please wait for the program to finish...")
    stop_event.set()  # Signal threads to stop
    if not HEADLESS and not ENABLE_WEBSERVER:
        cv2.destroyAllWindows()
    sys.exit(0)

# Making sure the data is compliant
def sanitize_detection_data(detection):
    return {
        'frame_count': int(detection['frame_count']),
        'timestamp': str(detection['timestamp']),
        'coordinates': [int(coord) for coord in detection['coordinates']],
        'confidence': float(detection['confidence'])
    }

# Main
if __name__ == "__main__":
    log_cuda_info()  # Log CUDA information
    parser = argparse.ArgumentParser(description="YOLOv8 RTMP Stream Human Detection")

    # Define mutually exclusive group for save_detections
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save_detections", action='store_true', help="Save images with detections")
    save_group.add_argument("--no_save_detections", action='store_false', dest='save_detections', help="Do not save images with detections")

    # Set default to None to indicate no override
    parser.set_defaults(save_detections=None)

    # Define other arguments without default values to allow config.ini to set them
    parser.add_argument("--headless", action='store_true', help="Run in headless mode without GUI display")
    parser.add_argument("--stream_url", type=str, help="URL of the RTMP stream")
    parser.add_argument("--use_webcam", action='store_true', help="Use webcam for video input")
    parser.add_argument("--webcam_index", type=int, help="Index number of the webcam to use")
    parser.add_argument("--conf_threshold", type=float, help="Confidence threshold for detections")
    parser.add_argument("--model_variant", type=str, help="YOLOv8 model variant to use")
    parser.add_argument("--image_format", type=str, help="Image format for saving detections (jpg or png)")
    parser.add_argument("--save_dir", type=str, help="Directory to save detection images")
    parser.add_argument("--retry_delay", type=int, help="Delay between retries to connect to the stream")
    parser.add_argument("--max_retries", type=int, help="Maximum number of retries to connect to the stream")
    parser.add_argument("--rescale_input", action='store_true', help="Rescale the input frames")
    parser.add_argument("--target_height", type=int, help="Target height for rescaling the frames")
    parser.add_argument("--denoise", action='store_true', help="Toggle denoising on/off")
    parser.add_argument("--process_fps", type=int, help="Frames per second for processing")
    parser.add_argument("--use_process_fps", action='store_true', help="Whether to use custom processing FPS")
    parser.add_argument("--timeout", type=int, help="Timeout for the video stream in seconds")
    parser.add_argument("--enable_webserver", action='store_true', help="Enable the web server for streaming detections")
    parser.add_argument("--webserver_host", type=str, help="Host IP address for the web server")
    parser.add_argument("--webserver_port", type=int, help="Port number for the web server")

    args = parser.parse_args()

    # Override configurations based on command-line arguments
    if args.enable_webserver:
        ENABLE_WEBSERVER = True
    if args.webserver_host:
        WEBSERVER_HOST = args.webserver_host
    if args.webserver_port:
        WEBSERVER_PORT = args.webserver_port

    # Override HEADLESS if --headless is specified
    if args.headless or ENABLE_WEBSERVER:
        HEADLESS = True
        main_logger.info("Running in headless mode -- no GUI. Set 'headless' AND 'enable_webserver' to 'false' if you want to run the windowed GUI version.")

    # Set SAVE_DETECTIONS based on args, with fallback to config
    # After parsing arguments
    if args.save_detections is not None:
        SAVE_DETECTIONS = args.save_detections
        main_logger.info(f"save_detections overridden by command-line argument: {SAVE_DETECTIONS}")
    else:
        SAVE_DETECTIONS = config.getboolean('detection', 'save_detections')
        main_logger.info(f"save_detections set from config.ini: {SAVE_DETECTIONS}")

    # # Similarly, set DRAW_RECTANGLES based on args, with fallback to config
    # if args.draw_rectangles is not None:
    #     DRAW_RECTANGLES = args.draw_rectangles
    # else:
    DRAW_RECTANGLES = config.getboolean('detection', 'draw_rectangles')
    main_logger.info(f"Draw rectangles set to: {DRAW_RECTANGLES}")

    # Update other configurations based on arguments, only if provided
    if args.stream_url:
        STREAM_URL = args.stream_url
    if args.use_webcam:
        USE_WEBCAM = args.use_webcam
    if args.webcam_index is not None:
        WEBCAM_INDEX = args.webcam_index
    if args.conf_threshold is not None:
        DEFAULT_CONF_THRESHOLD = args.conf_threshold
    if args.model_variant:
        DEFAULT_MODEL_VARIANT = args.model_variant
    if args.image_format:
        IMAGE_FORMAT = args.image_format
    if args.save_dir:
        SAVE_DIR = args.save_dir
    if args.retry_delay is not None:
        RETRY_DELAY = args.retry_delay
    if args.max_retries is not None:
        MAX_RETRIES = args.max_retries
    if args.rescale_input:
        RESCALE_INPUT = args.rescale_input
    if args.target_height is not None:
        TARGET_HEIGHT = args.target_height
    if args.denoise:
        DENOISE = args.denoise
    if args.process_fps is not None:
        PROCESS_FPS = args.process_fps
    if args.use_process_fps:
        USE_PROCESS_FPS = args.use_process_fps
    if args.timeout is not None:
        TIMEOUT = args.timeout

    # Initialize web server if enabled
    if ENABLE_WEBSERVER:
        # Start the web server in a separate daemon thread
        web_server_thread = threading.Thread(
            target=start_web_server,
            args=(
                WEBSERVER_HOST,
                WEBSERVER_PORT,
                detection_log_path,                
                detections_list,
                logs_list,
                detections_lock,
                logs_lock,
                config
                # detection_timestamps,  # Pass detection_timestamps
                # timestamps_lock        # Pass timestamps_lock
            ),
            daemon=True  # Set as daemon
        )
        web_server_thread.start()
        main_logger.info(f"Web server started at http://{WEBSERVER_HOST}:{WEBSERVER_PORT}")

    # Initialize SAVE_DIR using the updated get_save_dir function
    try:
        SAVE_DIR = get_current_save_dir()
        main_logger.info(f"Saving detections to directory: {SAVE_DIR}")
    except RuntimeError as e:
        main_logger.error(f"Error initializing save directory: {e}")
        sys.exit(1)

    # Initialize queues and events
    frame_queue = Queue(maxsize=10)
    stop_event = Event()
    detection_ongoing = Event()

    # Register signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize threads
    capture_thread = Thread(target=frame_capture_thread, args=(STREAM_URL, USE_WEBCAM, WEBCAM_INDEX, frame_queue, stop_event))
    processing_thread = Thread(target=frame_processing_thread, args=(frame_queue, stop_event, DEFAULT_CONF_THRESHOLD, DRAW_RECTANGLES, DENOISE, PROCESS_FPS, USE_PROCESS_FPS, detection_ongoing, HEADLESS))

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
        if not HEADLESS and not ENABLE_WEBSERVER:
            cv2.destroyAllWindows()
        main_logger.info("Program exited cleanly.")
