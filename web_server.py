# web_server.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Web server module for real-time YOLOv8 detection
# https://github.com/FlyingFathead/dvr-yolov8-detection/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Version number
import version  # Import the version module
version_number = version.version_number

import uuid
import sys
import subprocess
import os
import signal
import shutil

from collections import deque, OrderedDict
from datetime import datetime, date
import threading
import time
from flask import Flask, Response, render_template_string, stream_with_context, request, jsonify
from flask import send_from_directory
from flask import send_file
from werkzeug.utils import safe_join

# threaded serving via waitress
from waitress import serve 

import cv2
import logging
from web_graph import generate_detection_graph
import configparser
import json

# aggergation for the detections for webUI
from collections import defaultdict

# Configure logging for the web server
logger = logging.getLogger('web_server')
logger.setLevel(logging.INFO)
# Prevent messages from propagating to the root logger
logger.propagate = False
# init flask
app = Flask(__name__)

# Flask proxy fix
# from werkzeug.middleware.proxy_fix import ProxyFix
# app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
# app.config['APPLICATION_ROOT'] = '/'

# Global variables to hold the output frame and a lock for thread safety
output_frame = None
frame_lock = threading.Lock()
# HLS process status
hls_process = None 

# detect interrupt signals and safely write detections if enabled
def signal_handler(sig, frame):
    logger.info("Interrupt received, stopping the web server.")
    if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
        save_aggregated_detections()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_config(config_file='config.ini'):
    """Loads configuration from the specified INI file."""
    # config = configparser.ConfigParser()
    # Turn off interpolation so “%03d” won't cause a syntax error:
    config = configparser.ConfigParser(interpolation=None)    
    read_files = config.read(config_file)
    if not read_files:
        logger.error(f"Configuration file '{config_file}' not found or is empty.")
        # Optionally, you can raise an exception or set default configurations here
    return config

# Load configurations
config = load_config()

# Extract logging directory and files from the config
log_directory = config.get('logging', 'log_directory', fallback='./logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)  # Ensure log directory exists

log_file = os.path.join(log_directory, config.get('logging', 'log_file', fallback='logging.log'))
detection_log_file = os.path.join(log_directory, config.get('logging', 'detection_log_file', fallback='detections.log'))
access_log_file = os.path.join(log_directory, config.get('logging', 'access_log_file', fallback='access.log'))

# Configure the access logger
access_log_handler = logging.FileHandler(access_log_file)
access_log_formatter = logging.Formatter('%(asctime)s - %(message)s')
access_log_handler.setFormatter(access_log_formatter)
access_logger = logging.getLogger('access_logger')
access_logger.addHandler(access_log_handler)
access_logger.setLevel(logging.INFO)

# Dictionary to store timestamps of recent requests for each IP
last_logged_time = {}
log_interval = 10  # Interval in seconds to aggregate logs for frequent requests

# Extract configurations with fallbacks
ENABLE_WEBSERVER = config.getboolean('webserver', 'enable_webserver', fallback=True)
WEBSERVER_HOST = config.get('webserver', 'webserver_host', fallback='0.0.0.0')
WEBSERVER_PORT = config.getint('webserver', 'webserver_port', fallback=5000)
WEBSERVER_MAX_FPS = config.getint('webserver', 'webserver_max_fps', fallback=10)
MJPEG_QUALITY = config.getint('webserver', 'mjpeg_quality', fallback=75) # Read MJPEG quality for web preview
WEBUI_COOLDOWN_AGGREGATION = config.getint('webui', 'webui_cooldown_aggregation', fallback=30)
WEBUI_BOLD_THRESHOLD = config.getint('webui', 'webui_bold_threshold', fallback=10)
# Read check_interval from config.ini with a fallback to 10
interval_checks = config.getboolean('webserver', 'interval_checks', fallback=True)
check_interval = config.getint('webserver', 'check_interval', fallback=10)
# Persistent aggregated detections
ENABLE_PERSISTENT_AGGREGATED_DETECTIONS = config.getboolean('aggregation', 'enable_persistent_aggregated_detections', fallback=False)
AGGREGATED_DETECTIONS_FILE = config.get('aggregation', 'aggregated_detections_file', fallback='./logs/aggregated_detections.json')

# set the preview method and inform about it
preview_method = config.get('webserver', 'preview_method', fallback='mjpeg')
logger.info(f"Preview method set to: {preview_method}")
if preview_method == 'mjpeg':
    logger.info(f"MJPEG Quality set to: {MJPEG_QUALITY}") # Log the quality

if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
    logger.info(f"Persistent aggregated detections enabled. Logging to file: {AGGREGATED_DETECTIONS_FILE}")

# Gglobal variables for aggregation
# aggregated_detections_list = deque(maxlen=100)  # Adjust maxlen as needed

# get HLS directory if in use
HLS_OUTPUT_DIR = config.get('hls', 'hls_output_dir', fallback='/tmp/hls')

# Initialize the aggregated_detections_list and aggregated_lock
if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
    try:
        with open(AGGREGATED_DETECTIONS_FILE, 'r') as f:
            data = json.load(f)
            aggregated_detections_list = deque(data, maxlen=100)
            logger.info("Loaded aggregated detections from persistent storage.")
    except FileNotFoundError:
        logger.info("Aggregated detections file not found. Starting with an empty list.")
        aggregated_detections_list = deque(maxlen=100)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from aggregated detections file: {e}")
        aggregated_detections_list = deque(maxlen=100)
else:
    aggregated_detections_list = deque(maxlen=100)

aggregated_lock = threading.Lock()

# // client tracking
# Keep track of connected clients
connected_clients = {}

# Initialize the lock for connected_clients
connected_clients_lock = threading.Lock()

# on-request HLS streaming
def start_hls_ffmpeg_if_needed():
    global hls_process
    # If already running, do nothing
    if hls_process is not None:
        return

    # Pull config from app or global config
    hls_output_dir = app.config.get('hls_output_dir', '/tmp/hls')
    input_stream   = app.config.get('stream_url', 'rtmp://127.0.0.1:1935/live/stream')
    hls_time       = app.config.get('hls_time', '2')
    hls_list_size  = app.config.get('hls_list_size', '10')
    segment_pattern = app.config.get('segment_pattern', 'segment_%03d.ts')
    playlist_filename = app.config.get('playlist_filename', 'playlist.m3u8')

    # Ensure output directory exists
    os.makedirs(hls_output_dir, exist_ok=True)

    segment_path  = os.path.join(hls_output_dir, segment_pattern)
    playlist_path = os.path.join(hls_output_dir, playlist_filename)

    cmd = [
        'ffmpeg',
        '-i', input_stream,
        '-c', 'copy',
        '-f', 'hls',
        '-hls_time', str(hls_time),
        '-hls_list_size', str(hls_list_size),
        '-hls_segment_filename', segment_path,
        playlist_path
    ]
    logger.info("Starting HLS ffmpeg: " + " ".join(cmd))
    hls_process = subprocess.Popen(cmd)

# on-request hls stopper
def stop_hls_ffmpeg():
    global hls_process
    if hls_process is not None:
        logger.info("Terminating HLS ffmpeg...")
        hls_process.terminate()
        hls_process.wait(timeout=5)
        hls_process = None    

# rotate aggregated files
def rotate_aggregated_files():
    """Rotates the aggregated detections files when they exceed the max size."""
    base_file = AGGREGATED_DETECTIONS_FILE
    keep_old = config.getboolean('aggregation', 'keep_old_aggregations', fallback=True)
    max_old = config.getint('aggregation', 'max_old_aggregations', fallback=5)

    if not keep_old:
        os.remove(base_file)
        logger.info("Old aggregated detections file removed.")
        return

    # Rotate files
    for i in range(max_old, 0, -1):
        old_file = f"{base_file}.{i}"
        if os.path.exists(old_file):
            if i == max_old:
                os.remove(old_file)
            else:
                new_file = f"{base_file}.{i+1}"
                os.rename(old_file, new_file)
    # Rename the current file to .1
    if os.path.exists(base_file):
        os.rename(base_file, f"{base_file}.1")
    logger.info("Aggregated detections file rotated.")

# Save aggregated detections if enabled
def save_aggregated_detections():
    """Saves the aggregated detections to a JSON file."""
    if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
        with aggregated_lock:
            data = list(aggregated_detections_list)
        try:
            # Write the data to the aggregated detections file
            with open(AGGREGATED_DETECTIONS_FILE, 'w') as f:
                json.dump(data, f, default=str)
            logger.info("Aggregated detections saved to persistent storage.")
        except Exception as e:
            logger.error(f"Error saving aggregated detections to file: {e}")

# Periodically check and log active client connections.
def log_active_connections():
    """Periodically log active client connections and remove inactive ones."""
    try:
        logger.info("Starting active connections logging thread.")  # Confirmation log
        previous_clients = set()
        timeout = 60  # seconds
        while True:
            time.sleep(check_interval)
            current_time = datetime.now()
            current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

            with connected_clients_lock:
                active_clients = set(connected_clients.keys())

                # Identify inactive clients
                inactive_ips = []
                for ip, last_seen in connected_clients.items():
                    if (current_time - last_seen).total_seconds() > timeout:
                        inactive_ips.append(ip)

                # Remove inactive clients
                for ip in inactive_ips:
                    del connected_clients[ip]
                    logger.info(f"Removed inactive client: {ip}")

                # Update active_clients after removals
                active_clients = set(connected_clients.keys())

                if active_clients != previous_clients:  # Log only when there's a change
                    if active_clients:
                        logger.info(f"Active connections at {current_time_str}: {', '.join(active_clients)}")
                    else:
                        logger.info(f"No active web UI connections at {current_time_str}, check interval is {check_interval} seconds.")

                    previous_clients = active_clients.copy()  # Update to avoid redundant logging
                else:
                    logger.debug(f"No change in active connections at {current_time_str}.")
    except Exception as e:
        logger.error(f"Error in log_active_connections thread: {e}")

# # Periodically check and log active client connections.
# def log_active_connections():
#     """Periodically log active client connections."""
#     logger.info("Starting active connections logging thread.")  # Add this log to confirm thread starts
#     previous_clients = set()
#     while True:
#         time.sleep(check_interval)
#         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#         with connected_clients_lock:
#             active_clients = set(connected_clients.keys())

#             if active_clients != previous_clients:  # Only log when there's a change in active connections
#                 if active_clients:
#                     logger.info(f"Active connections at {current_time}: {connected_clients}")
#                 else:
#                     logger.info(f"No active web UI connections at {current_time}, check interval is {check_interval} seconds.")
                
#                 previous_clients = active_clients.copy()  # Update previous clients to avoid redundant logging

# Conditionally start the background thread for logging active connections
if interval_checks:
    threading.Thread(target=log_active_connections, daemon=True).start()
    logger.info("Active connections logging is enabled.")
else:
    logger.info("Active connections logging is disabled.")

def start_web_server(host='0.0.0.0', port=5000, detection_log_path=None,
                     detections_list=None, logs_list=None, detections_lock=None,
                     logs_lock=None, config=None, save_dir_base=None):
    """Starts the Flask web server."""

    app.config['config'] = config  # Store config in Flask's config if needed

    # Initialize SAVE_DIR_BASE within the web server process
    # SAVE_DIR_BASE = get_base_save_dir(config)
    # app.config['SAVE_DIR_BASE'] = SAVE_DIR_BASE

    if save_dir_base:
        SAVE_DIR_BASE = save_dir_base
        logger.info(f"Using SAVE_DIR_BASE passed from detection script: {SAVE_DIR_BASE}")
    else:
        # Initialize SAVE_DIR_BASE within the web server process
        SAVE_DIR_BASE = get_base_save_dir(config)
        logger.info(f"Initialized SAVE_DIR_BASE within web server: {SAVE_DIR_BASE}")

    app.config['mjpeg_quality'] = config.getint('webserver', 'mjpeg_quality', fallback=75)

    app.config['SAVE_DIR_BASE'] = SAVE_DIR_BASE

    logger.info(f"SAVE_DIR_BASE is set to: {app.config['SAVE_DIR_BASE']}")
    
    logger.info(f"Starting web server at http://{host}:{port}")
    app.config['detection_log_path'] = detection_log_path
    app.config['detections_list'] = detections_list if detections_list is not None else []
    app.config['logs_list'] = logs_list if logs_list is not None else []
    app.config['detections_lock'] = detections_lock if detections_lock is not None else threading.Lock()
    app.config['logs_lock'] = logs_lock if logs_lock is not None else threading.Lock()
    # Access the max FPS from the config
    app.config['webserver_max_fps'] = config.getint('webserver', 'webserver_max_fps', fallback=10)
    # Suppress Flask's default logging if necessary
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    # Start the aggregation thread here
    aggregation_thread = threading.Thread(
        target=aggregation_thread_function,
        args=(
            app.config['detections_list'],
            app.config['detections_lock'],
            WEBUI_COOLDOWN_AGGREGATION,
            WEBUI_BOLD_THRESHOLD
        ),
        daemon=True
    )
    aggregation_thread.start()
    logger.info("Aggregation thread started.")

    # Log the active configurations on startup
    logger.info("======================================================")
    logger.info("Web Server Configuration:")
    logger.info(f"Enable Web Server: {config.getboolean('webserver', 'enable_webserver', fallback=True)}")
    logger.info(f"Web Server Host: {host}")
    logger.info(f"Web Server Port: {port}")
    logger.info(f"Web Server Max FPS: {app.config['webserver_max_fps']}")
    logger.info(f"Preview Method: {preview_method}")
    if preview_method == 'mjpeg':
        logger.info(f"MJPEG Stream Quality: {app.config['mjpeg_quality']}")    
    logger.info(f"Check Interval: {config.getint('webserver', 'check_interval', fallback=10)} seconds")
    logger.info(f"Web UI Cooldown Aggregation: {config.getint('webui', 'webui_cooldown_aggregation', fallback=30)} seconds")
    logger.info(f"Web UI Bold Threshold: {config.getint('webui', 'webui_bold_threshold', fallback=10)}")
    logger.info(f"Persistent Aggregated Detections Enabled: {ENABLE_PERSISTENT_AGGREGATED_DETECTIONS}")
    if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
        logger.info(f"Aggregated Detections File: {AGGREGATED_DETECTIONS_FILE}")    
    logger.info(f"SAVE_DIR_BASE is set to: {app.config['SAVE_DIR_BASE']}")
    logger.info("======================================================")

    # app.run(host=host, port=port, threaded=True)
    # // old method
    # app.run(host=host, port=port, threaded=True, use_reloader=False)    
    serve(app, host=host, port=port, threads=8)

def set_output_frame(frame):
    """Updates the global output frame to be served to clients."""
    global output_frame
    with frame_lock:
        output_frame = frame.copy()

def generate_frames():
    """Generator function that yields frames in byte format for streaming."""
    global output_frame
    max_fps = app.config.get('webserver_max_fps', 10)
    # --- ADDED: Get MJPEG Quality from app.config ---
    mjpeg_quality = app.config.get('mjpeg_quality', 75) # Default to 75 if not found
    # --- END ADDED ---

    frame_interval = 1.0 / max_fps if max_fps > 0 else 0 # Allow 0 FPS to effectively disable limit
    last_frame_time = time.time()

    client_ip = request.remote_addr
    logger.info(f"MJPEG Client Connected: {client_ip}. Max FPS: {max_fps}, Quality: {mjpeg_quality}")

    while True:
        frame_copy = None
        with frame_lock:
            if output_frame is not None:
                frame_copy = output_frame.copy()

        if frame_copy is None:
            time.sleep(0.05) # Wait if no frame available yet
            continue

        # --- FPS Limiting (applied BEFORE encoding) ---
        current_time = time.time()
        elapsed_since_last = current_time - last_frame_time
        if frame_interval > 0 and elapsed_since_last < frame_interval:
             wait_time = frame_interval - elapsed_since_last
             time.sleep(wait_time)
        # Update time *after* potential sleep, *before* encoding
        last_frame_time = time.time()
        # --- End FPS Limiting ---

        # Encode the frame in JPEG format *using the configured quality*
        try:
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, mjpeg_quality]
            ret, jpeg_buffer = cv2.imencode('.jpg', frame_copy, encode_param)
            if not ret:
                logger.warning("cv2.imencode failed, skipping frame.")
                continue # Skip this frame if encoding fails
            frame_bytes = jpeg_buffer.tobytes()
        except Exception as e:
             logger.error(f"Error during cv2.imencode: {e}")
             continue # Skip frame on encoding error

        # Yield the output frame in byte format
        try:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            logger.info(f"MJPEG Client Disconnected (GeneratorExit): {client_ip}")
            break # Client disconnected
        except Exception as e:
            logger.error(f"Error yielding MJPEG frame to {client_ip}: {e}")
            break # Other error during yield

# aggergation thread function

# Inside aggregation_thread_function
def aggregation_thread_function(detections_list, detections_lock, cooldown=30, bold_threshold=10):
    global aggregated_detections_list, aggregated_lock  # Declare globals
    last_detection_time = None
    current_aggregation = None  # To track ongoing aggregation

    while True:
        try:
            time.sleep(1)  # Check every second

            new_detections = []
            with detections_lock:
                if detections_list:
                    new_detections.extend(detections_list)
                    detections_list.clear()

            if new_detections:
                for detection in new_detections:
                    timestamp = datetime.strptime(detection['timestamp'], '%Y-%m-%d %H:%M:%S')
                    confidence = detection['confidence']
                    image_info = detection.get('image_filenames', {})

                    # Extract image filenames if they exist
                    image_filename_entry = {}
                    if 'full_frame' in image_info:
                        image_filename_entry['full_frame'] = image_info['full_frame']
                    if 'detection_area' in image_info:
                        image_filename_entry['detection_area'] = image_info['detection_area']

                    if current_aggregation is None:
                        # Start a new aggregation - ADD UUID HERE
                        aggregation_uuid = str(uuid.uuid4()) # Generate a unique ID
                        current_aggregation = {
                            'uuid': aggregation_uuid, # Store the UUID
                            'count': 1,
                            'first_timestamp': timestamp,
                            'latest_timestamp': timestamp,
                            'lowest_confidence': confidence,
                            'highest_confidence': confidence,
                            'image_filenames': [image_filename_entry] if image_filename_entry else [],
                            'finalized': False
                        }
                        with aggregated_lock:
                            # Prepend the current aggregation to the list
                            aggregated_detections_list.appendleft(current_aggregation)
                            logger.debug(f"Started new aggregation with UUID: {aggregation_uuid}") # Optional logging
                    else:
                        # Update the ongoing aggregation
                        current_aggregation['count'] += 1
                        current_aggregation['latest_timestamp'] = max(current_aggregation['latest_timestamp'], timestamp)
                        current_aggregation['lowest_confidence'] = min(current_aggregation['lowest_confidence'], confidence)
                        current_aggregation['highest_confidence'] = max(current_aggregation['highest_confidence'], confidence)
                        if image_filename_entry:
                            current_aggregation['image_filenames'].append(image_filename_entry)

                    last_detection_time = time.time()

                # Update the summary of the current aggregation
                update_aggregation_summary(current_aggregation, cooldown, bold_threshold)

                # Save potentially updated aggregation (including new images)
                if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
                    save_aggregated_detections() # Save on updates too

            # Check if cooldown period has passed since the last detection
            if current_aggregation and last_detection_time and (time.time() - last_detection_time) >= cooldown:
                # Mark current aggregation as finalized
                current_aggregation['finalized'] = True
                logger.debug(f"Finalized aggregation with UUID: {current_aggregation.get('uuid', 'N/A')}") # Optional logging
                current_aggregation = None
                last_detection_time = None

                # Save finalized state
                if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
                    save_aggregated_detections()

        except Exception as e:
            logger.error(f"Error in aggregation_thread_function: {e}", exc_info=True)

# # // old aggregation thread function
# def aggregation_thread_function(detections_list, detections_lock, cooldown=30, bold_threshold=10):
#     global aggregated_detections_list, aggregated_lock  # Declare globals
#     last_detection_time = None
#     current_aggregation = None  # To track ongoing aggregation

#     while True:
#         try:
#             time.sleep(1)  # Check every second

#             new_detections = []
#             with detections_lock:
#                 if detections_list:
#                     new_detections.extend(detections_list)
#                     detections_list.clear()

#             if new_detections:
#                 for detection in new_detections:
#                     timestamp = datetime.strptime(detection['timestamp'], '%Y-%m-%d %H:%M:%S')
#                     confidence = detection['confidence']
#                     image_info = detection.get('image_filenames', {})

#                     # Extract image filenames if they exist
#                     image_filename_entry = {}
#                     if 'full_frame' in image_info:
#                         image_filename_entry['full_frame'] = image_info['full_frame']
#                     if 'detection_area' in image_info:
#                         image_filename_entry['detection_area'] = image_info['detection_area']

#                     if current_aggregation is None:
#                         # Start a new aggregation
#                         current_aggregation = {
#                             'count': 1,
#                             'first_timestamp': timestamp,
#                             'latest_timestamp': timestamp,
#                             'lowest_confidence': confidence,
#                             'highest_confidence': confidence,
#                             'image_filenames': [image_filename_entry] if image_filename_entry else [],
#                             'finalized': False
#                         }
#                         with aggregated_lock:
#                             # Prepend the current aggregation to the list
#                             aggregated_detections_list.appendleft(current_aggregation)
#                     else:
#                         # Update the ongoing aggregation
#                         current_aggregation['count'] += 1
#                         current_aggregation['latest_timestamp'] = max(current_aggregation['latest_timestamp'], timestamp)
#                         current_aggregation['lowest_confidence'] = min(current_aggregation['lowest_confidence'], confidence)
#                         current_aggregation['highest_confidence'] = max(current_aggregation['highest_confidence'], confidence)
#                         if image_filename_entry:
#                             current_aggregation['image_filenames'].append(image_filename_entry)

#                     last_detection_time = time.time()

#                 # Update the summary of the current aggregation
#                 update_aggregation_summary(current_aggregation, cooldown, bold_threshold)

#                 if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
#                     save_aggregated_detections()

#             # Check if cooldown period has passed since the last detection
#             if current_aggregation and last_detection_time and (time.time() - last_detection_time) >= cooldown:
#                 # Mark current aggregation as finalized
#                 current_aggregation['finalized'] = True
#                 current_aggregation = None
#                 last_detection_time = None

#                 if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
#                     save_aggregated_detections()

#         except Exception as e:
#             logger.error(f"Error in aggregation_thread_function: {e}", exc_info=True)

def update_aggregation_summary(current_aggregation, cooldown, bold_threshold):
    count_display = f"<strong>{current_aggregation['count']}</strong>" if current_aggregation['count'] >= bold_threshold else f"{current_aggregation['count']}"
    summary = (
        f"👀 Human detected {count_display} times within {cooldown} seconds! "
        f"First seen: {current_aggregation['first_timestamp']:%Y-%m-%d %H:%M:%S}, "
        f"Latest: {current_aggregation['latest_timestamp']:%Y-%m-%d %H:%M:%S}, "
        f"Lowest confidence: {current_aggregation['lowest_confidence']:.2f}, "
        f"Highest confidence: {current_aggregation['highest_confidence']:.2f}"
    )
    current_aggregation['summary'] = summary

def finalize_aggregation(current_aggregation, cooldown, bold_threshold):
    global aggregated_detections_list, aggregated_lock  # Declare globals
    count_display = f"<strong>{current_aggregation['count']}</strong>" if current_aggregation['count'] >= bold_threshold else f"{current_aggregation['count']}"
    summary = (
        f"👀 Human detected {count_display} times within {cooldown} seconds! "
        f"First seen: {current_aggregation['first_timestamp']:%Y-%m-%d %H:%M:%S}, "
        f"Latest: {current_aggregation['latest_timestamp']:%Y-%m-%d %H:%M:%S}, "
        f"Lowest confidence: {current_aggregation['lowest_confidence']:.2f}, "
        f"Highest confidence: {current_aggregation['highest_confidence']:.2f}"
    )

    with aggregated_lock:
        # Prepend the finalized aggregation to the list
        aggregated_detections_list.appendleft({
            'summary': summary,
            'image_filenames': current_aggregation['image_filenames']
        })
    logger.info("Aggregation finalized and added to the list.")

    if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
        save_aggregated_detections()

# get the candidates for save directories; use fallbacks when needed
def get_save_dir_candidates(config):
    """
    Gets a list of candidate base directories for searching detection images,
    in the same order of preference as the detection script.
    """
    candidates = []
    
    use_env = config.getboolean('detection', 'use_env_save_dir', fallback=False)
    env_var = config.get('detection', 'env_save_dir_var', fallback='YOLO_SAVE_DIR')
    
    if use_env:
        env_dir = os.getenv(env_var)
        if env_dir:
            candidates.append(env_dir)

    # The order MUST match the logic in yolov8_live_rtmp_stream_detection.py
    # 1. Primary directory
    primary_dir = config.get('detection', 'default_save_dir', fallback=None)
    if primary_dir:
        candidates.append(primary_dir)

    # 2. Fallback directory
    fallback_dir = config.get('detection', 'fallback_save_dir', fallback=None)
    if fallback_dir:
        candidates.append(fallback_dir)

    # 3. 'save_dir_base' as another fallback
    save_dir_base = config.get('detection', 'save_dir_base', fallback=None)
    if save_dir_base and save_dir_base not in candidates:
        candidates.append(save_dir_base)
    
    logger.debug(f"Web server will search for images in these directories: {candidates}")
    return candidates

@app.before_request
def log_request_info():
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    current_time = time.time()

    # List of endpoints to completely ignore logging
    excluded_routes = ['/api/current_time', '/api/detections', '/api/logs', '/video_feed', '/static/', '/favicon.ico']

    # Track IP addresses for active connections, regardless of the route
    with connected_clients_lock:
        # connected_clients[client_ip] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        connected_clients[client_ip] = datetime.now()

    # If the current request path is in the excluded routes, skip logging
    if any(request.path.startswith(route) for route in excluded_routes):
        return

    # Get the last logged time for this client
    last_time = last_logged_time.get(client_ip, 0)

    # Log requests only if enough time has passed
    if current_time - last_time > log_interval:
        access_logger.info(f"Client IP: {client_ip} - Request URL: {request.url} - Request Path: {request.path} - Request Headers: {request.headers} - Method: {request.method} - User Agent: {request.user_agent}")
        last_logged_time[client_ip] = current_time

# @app.before_request
# def log_request_info():
#     client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
#     current_time = time.time()

#     # Log request details into access log
#     # access_logger.info(f"Client IP: {client_ip} - Request URL: {request.url} - Method: {request.method} - User Agent: {request.user_agent}")
    
#     # List of endpoints to ignore logging (e.g., video feed spam)
#     # excluded_routes = ['/video_feed', '/static/', '/favicon.ico']
#     # excluded_routes = "/api/current_time"
#     excluded_routes = ['/api/current_time', '/api/detections']

#     # Optional: Add this if you need more detailed logs in the main log
#     # logging.info("⚠️ User connected to the webUI:")
#     # logging.info(f"Request path: {request.path}")
#     # logging.info(f"Request headers: {request.headers}")

#     # Track when clients hit an endpoint
#     if request.path not in excluded_routes:
#         with connected_clients_lock:
#             connected_clients[client_ip] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#     # Log requests for non-excluded routes immediately
#     if request.path not in excluded_routes:
#         access_logger.info(f"Client IP: {client_ip} - Request URL: {request.url} - Request Path: {request.path} - Request Headers: {request.headers} - Method: {request.method} - User Agent: {request.user_agent}")
#     else:
#         # Check if enough time has passed to log this IP again
#         last_time = last_logged_time.get(client_ip, 0)
#         if current_time - last_time > log_interval:
#             access_logger.info(f"Aggregated log - Client IP: {client_ip} - Request Path: {request.path} - Request Headers: {request.headers} - Request URL: {request.url} - Method: {request.method} - User Agent: {request.user_agent}")
#             last_logged_time[client_ip] = current_time

@app.route('/api/', methods=['GET'])
def api_root():
    return jsonify({"message": "API Root"}), 200
      
@app.route('/api/detections/<path:filename>')
def serve_detection_image(filename):
    config = app.config.get('config')
    if not config:
        logger.error("Config not found in app context. Cannot determine save directories.")
        return "Internal Server Error", 500

    search_dirs = get_save_dir_candidates(config)
    logger.info(f"Searching for '{filename}' in candidate directories: {search_dirs}")

    for base_dir in search_dirs:
        try:
            # Use os.path.abspath to resolve any relative paths like '.' or '..'
            abs_base_dir = os.path.abspath(base_dir)
            # Join the absolute base with the relative filename
            filepath = os.path.join(abs_base_dir, filename)
            
            # Security check: After joining, the final absolute path must still be inside the base directory.
            # This is a robust way to prevent directory traversal attacks (e.g., filename being '../.../etc/passwd')
            if not os.path.abspath(filepath).startswith(abs_base_dir):
                logger.warning(f"Path traversal attempt blocked: '{filename}' from base '{base_dir}'")
                continue

            if os.path.isfile(filepath):
                logger.info(f"Found and serving file: {filepath}")
                return send_file(filepath)
            
        except Exception as e:
            logger.error(f"Error when checking path in '{base_dir}': {e}")
            continue
    
    logger.error(f"File '{filename}' not found in any of the search directories.")
    return "File not found", 404

    
@app.route('/api/toggle_preview', methods=['POST'])
def toggle_preview():
    """Toggle between MJPEG and HLS preview on demand."""
    current_method = app.config.get('preview_method', 'mjpeg')

    if current_method == 'mjpeg':
        # Switch to HLS
        app.config['preview_method'] = 'hls'
        start_hls_ffmpeg_if_needed()
        logger.info("Switched preview_method to HLS.")
        return jsonify({"status": "ok", "new_method": "hls"}), 200
    else:
        # Switch to MJPEG
        app.config['preview_method'] = 'mjpeg'
        stop_hls_ffmpeg()
        logger.info("Switched preview_method to MJPEG.")
        return jsonify({"status": "ok", "new_method": "mjpeg"}), 200

# # send from directory method // flask doesn't allow
# @app.route('/detections/<path:filename>')
# def serve_detection_image(filename):
#     save_dir_base = app.config.get('SAVE_DIR_BASE', './yolo_detections')
#     filepath = safe_join(save_dir_base, filename)
#     if os.path.isfile(filepath):
#         directory = os.path.dirname(filepath)
#         filename_only = os.path.basename(filepath)
#         return send_from_directory(directory, filename_only)
#     else:
#         return "File not found", 404

@app.route('/api/current_time')
def get_current_time():
    """API endpoint to return the current host time."""
    host_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return jsonify({'current_time': host_time})

if preview_method == 'mjpeg':
    @app.route('/video_feed')
    def video_feed():
        """Video streaming route (MJPEG)."""
        return Response(
            stream_with_context(generate_frames()),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
else:
    logger.info("MJPEG route disabled (preview_method != 'mjpeg').")

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(stream_with_context(generate_frames()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame',
#                     headers={
#                         'Cache-Control': 'no-cache, no-store, must-revalidate',
#                         'Pragma': 'no-cache',
#                         'Expires': '0'
#                     })

# for HLS
@app.route('/hls/<path:filename>')
def hls_files(filename):
    """
    Serve the .m3u8 and .ts files from HLS_OUTPUT_DIR,
    so that /hls/playlist.m3u8 and /hls/segment_000.ts will be found.
    """
    return send_from_directory(HLS_OUTPUT_DIR, filename)

# get aggregated detections rather than flood the webui
@app.route('/api/detections')
def get_detections():
    # Read max_entries from config.ini with a fallback to 100
    max_entries = config.getint('aggregation', 'webui_max_aggregation_entries', fallback=100)
    with aggregated_lock:
        aggregated_detections = list(aggregated_detections_list)[:max_entries]
    response = jsonify(aggregated_detections)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# uuid image route
@app.route('/api/detection_images/<uuid_str>', methods=['GET'])
def get_detection_images(uuid_str):
    """API endpoint to get image filenames for a specific aggregated detection."""
    logger.debug(f"Request received for images of detection UUID: {uuid_str}")
    found_images = None
    with aggregated_lock:
        # Iterate through the deque to find the matching UUID
        for item in aggregated_detections_list:
            # Ensure the item has a UUID key before comparing
            if item.get('uuid') == uuid_str:
                found_images = item.get('image_filenames', []) # Get filenames or empty list
                break # Found it, no need to continue searching

    if found_images is not None:
        logger.debug(f"Found {len(found_images)} images for UUID: {uuid_str}")
        response = jsonify(found_images)
        # Prevent caching of this dynamic data
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    else:
        logger.warning(f"Detection with UUID {uuid_str} not found in aggregated list.")
        return jsonify({"error": "Detection not found"}), 404

# // (old method)
# @app.route('/api/detections')
# def get_detections():
#     with app.config['detections_lock']:
#         detections = list(app.config['detections_list'])

#     detections_data = []
#     for det in detections:
#         detection_info = {
#             'frame_count': int(det['frame_count']),
#             'timestamp': str(det['timestamp']),
#             'coordinates': [int(coord) for coord in det['coordinates']],
#             'confidence': float(det['confidence'])
#         }
#         detections_data.append(detection_info)

#     return jsonify(detections_data)
#     # Alternatively, return detections directly if they are already in the desired format
#     # return jsonify(detections)

def group_aggregations_by_date(aggregated_data):
    """
    Groups aggregator entries by date based on 'first_timestamp'.
    Returns an OrderedDict like:
        "Today (26. March 2026)" -> [list of aggregator items],
        "Yesterday (25. March 2026)" -> [...],
        "24. March 2026" -> [...], etc.
    Most recent day first (descending by date).
    """
    # First, transform any string timestamps into real datetime objects
    # so we can do date comparisons, sorting, etc.
    # We'll store them in a new list to avoid messing up the original.
    normalized = []
    for item in aggregated_data:
        # If we have a string timestamp, parse it. If it's already a datetime, keep it.
        # (The aggregator logic might store them as strings once loaded from JSON.)
        # We'll standardize on 'first_timestamp' for grouping.
        if isinstance(item['first_timestamp'], str):
            dt = datetime.strptime(item['first_timestamp'], "%Y-%m-%d %H:%M:%S")
        else:
            dt = item['first_timestamp']  # Already a datetime object
        # Make a shallow copy of the item plus a normalized date
        copy_item = dict(item)
        copy_item['first_timestamp'] = dt
        normalized.append(copy_item)

    # Sort them by 'first_timestamp' descending (newest first)
    sorted_data = sorted(normalized, key=lambda d: d['first_timestamp'], reverse=True)

    # We'll group them in an OrderedDict
    grouped = OrderedDict()

    # Helper to produce date headings
    def date_heading(dt: datetime):
        # Compare dt.date() to today's date
        day_date = dt.date()
        today_date = date.today()
        delta = (today_date - day_date).days

        if delta == 0:
            return f"Today ({day_date.strftime('%d. %B %Y')})"
        elif delta == 1:
            return f"Yesterday ({day_date.strftime('%d. %B %Y')})"
        else:
            return day_date.strftime("%d. %B %Y")

    # Group them up
    for item in sorted_data:
        dt_label = date_heading(item['first_timestamp'])
        if dt_label not in grouped:
            grouped[dt_label] = []
        grouped[dt_label].append(item)

    return grouped

@app.route('/api/logs')
def get_logs():
    with app.config['logs_lock']:
        logs = list(app.config['logs_list'])
    return jsonify(logs)

@app.route('/')
def index():
    base_path = request.script_root
    """Homepage to display video streaming and aggregated detections."""
    with aggregated_lock:
        detections = list(aggregated_detections_list)
    with app.config['logs_lock']:
        logs = list(app.config['logs_list'])

    # Group aggregator items by date
    grouped_detections = group_aggregations_by_date(detections)

    # Get the selected time range from query parameters
    hours = request.args.get('hours', default=None, type=int)
    graph_image = None
    if hours:
        detection_log_path = app.config['detection_log_path']
        graph_image = generate_detection_graph(hours, detection_log_path)

    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Human Detection</title>
    <!-- Removed the meta refresh tag -->
    <style>
        /* Basic styling */
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            position: relative; /* To position elements relative to the body */
        }
        h1, h2, h3 {
            color: #333;
        }
        ul {
            list-style-type: none;
            padding: 0;
        }
        li {
            background: #f4f4f4;
            margin: 5px 0;
            padding: 10px;
            border-radius: 4px;
        }
        #detection-graph {
            margin-top: 20px;
        }
        /* Styling for the time display */
        #time-container {
            position: fixed; /* Fixed position relative to the viewport */
            top: 20px;
            right: 20px;
            background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent background */
            padding: 10px 15px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            font-size: 1.1em;
            color: #333;
        }
        /* Modal styles */
        #image-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }

        #modal-content {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            z-index: 1000; /* Base z-index */                                  
        }

        #modal-image {
            user-drag: none;        /* Prevent dragging */
            user-select: none;      /* Prevent text selection */
            cursor: zoom-in;        /* Indicate clickable */
            
            /* === ADDED/MODIFIED CONSTRAINTS === */
            display: block;         /* Ensure it behaves like a block element */
            max-width: 90vw;        /* Max 90% of viewport width */
            max-height: 75vh;       /* Max 75% of viewport height (leaves room for controls) */
            object-fit: contain;    /* Scale image down to fit container, maintaining aspect ratio */
            /* === END ADDED/MODIFIED CONSTRAINTS === */

            margin-bottom: 10px;    /* Keep existing margin */
        }
                                  
        /* CSS for the X button */
        /* Adjust the close icon (X button) */
        #image-modal span {
            position: absolute;
            top: 2vh; /* Adjusted from '20px' */
            right: 2vh; /* Adjusted from '20px' */
            color: white;
            font-size: 5vh; /* Increased size using viewport height */
            cursor: pointer;
            z-index: 1003;
        }
                                  
        /* Modal image count */
        #image-count {
            bottom: 12vh; /* Adjusted from '80px' */
            left: 50%;
            transform: translateX(-50%);
            color: white;
            font-size: 2vh; /* Adjusted font size */
            text-align: center;
        }

        /* Modal button styles */
        #prev-button, #next-button {
            position: absolute;
            top: 0;
            width: 20%; /* Increased width for better tap area */
            height: 100%;
            background: transparent;
            border: none;
            cursor: pointer;
            outline: none;
            z-index: 1000;
        }

        #prev-button {
            left: 0;
        }

        #next-button {
            right: 0;
        }

        /* Add arrows using pseudo-elements */
        #prev-button::before, #next-button::before {
            content: '';
            position: absolute;
            top: 50%;
            width: 5vh; /* Adjusted from '30px' */
            height: 5vh; /* Adjusted from '30px' */
            margin-top: -2.5vh; /* Half of the new height */
            background-size: 5vh 5vh; /* Ensure the background image scales */
            background-repeat: no-repeat;
            background-position: center;
        }


        #prev-button::before {
            left: 2vh; /* Adjusted from '10px' */
            background-image: url('data:image/svg+xml;charset=UTF8,<svg xmlns="http://www.w3.org/2000/svg" fill="%23fff" viewBox="0 0 24 24"><path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg>');
        }

        #next-button::before {
            right: 2vh; /* Adjusted from '10px' */
            background-image: url('data:image/svg+xml;charset=UTF8,<svg xmlns="http://www.w3.org/2000/svg" fill="%23fff" viewBox="0 0 24 24"><path d="M8.59 16.59L10 18l6-6-6-6-1.41 1.41L13.17 12z"/></svg>');
        }

        /* Adjust the Swap button */
        #swap-button {
            position: absolute;
            bottom: 2vh; /* Adjusted from '20px' */
            left: 50%;
            transform: translateX(-50%);
            padding: 2vh 4vh; /* Adjusted for better size */
            font-size: 2.5vh; /* Increased font size */
            cursor: pointer;
            background-color: rgba(255, 255, 255, 0.9);
            border: none;
            border-radius: 5px;
            z-index: 1001;
        }

        #center-buttons #swap-button {
        position: static; /* Let the flex container handle positioning */
        transform: none;
        }

        /* Responsive design for mobile devices */
        @media only screen and (max-width: 600px) {
            #image-modal span {
                font-size: 6vh; /* Even larger on small screens */
            }
            #swap-button {
                font-size: 3vh;
                padding: 3vh 5vh;
            }
            #prev-button::before, #next-button::before {
                width: 6vh;
                height: 6vh;
                margin-top: -3vh;
                background-size: 6vh 6vh;
            }
            #image-count {
                font-size: 2.5vh;
            }
        }

        /* Center button container always fixed at the bottom */
        #center-buttons {
        position: fixed;
        bottom: 2vh;           /* 2% of viewport height from the bottom */
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        gap: 5px;              /* reduced gap between buttons */
        z-index: 1001;
        }

        /* All buttons in the center container get a base style */
        #center-buttons button {
        padding: 1.5vh 3vh;    /* a bit smaller padding */
        font-size: 2.5vh;
        background-color: rgba(255, 255, 255, 0.9);
        border: none;
        border-radius: 5px;
        cursor: pointer;
        }

        /* Make the skip buttons even slimmer */
        #skip-back-button,
        #skip-forward-button {
        padding: 1vh 2vh;
        font-size: 2vh;
        }                                  
                                  
        /* Loading spinner */
        #loading-spinner {
            display: none; /* Hidden by default */
            position: absolute;
            top: 50%;
            left: 50%;
            width: 40px;
            height: 40px;
            margin: -20px 0 0 -20px; /* Center the spinner */
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid #fff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            z-index: 1002; /* Ensure it's above the image but below the close button */
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Footer styles */
        footer {
            text-align: center;
            margin-top: 40px; /* Space above the footer */
        }
        footer hr {
            border: none;
            border-top: 1px solid #ccc;
            margin: 20px 0;
        }
        footer p {
            font-size: 0.8em; /* Small text */
            color: #666; /* Grey color for subtlety */
        }

    </style>

    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>                                    

</head>
<body>
                                  
    <!-- Time Display Container -->
    <div id="time-container">
        <strong>Current Host Time:</strong> <span id="host-time">Loading...</span>
    </div>

    <h1>Real-time Human Detection</h1>

<!---    <button id="preview-toggle-btn">Toggle MJPEG/HLS</button>
    <script>
    document.getElementById('preview-toggle-btn').addEventListener('click', function(e) {
    fetch('{{ base_path }}/api/toggle_preview', {
        method: 'POST'
    })
    .then(resp => resp.json())
    .then(data => {
        console.log('Toggle preview response:', data);
        // Option #1: reload the page so it picks up the new method
        location.reload();
        // Or #2: dynamically update video element:
        // ...
    })
    .catch(err => console.error('Error toggling preview:', err));
    });
    </script> --!>

    {% if preview_method == "mjpeg" %}
        <h2>(MJPEG Preview)</h2>
        <img src="{{ base_path }}{{ url_for('video_feed') }}" width="100%">
    {% else %}
        <h2>(HLS Preview)</h2>
        <!-- # We give the video an ID so we can attach Hls.js if needed: -->
        <video 
            id="hls-video" 
            controls 
            autoplay 
            muted 
            playsinline 
            width="100%"
        ></video>

        <script>
        // Quick snippet to attach HLS if not natively supported:
        const video = document.getElementById('hls-video');
        
        const hlsPlaylist = './hls/playlist.m3u8';
                                  
        // other variations
        // const basePath = "{{ base_path }}";
        // const hlsPlaylist = basePath + '/hls/playlist.m3u8';
        
        if (video.canPlayType('application/vnd.apple.mpegurl')) {
            // Safari, iOS, etc. can play HLS natively
            video.src = hlsPlaylist;
        } else if (Hls.isSupported()) {
            // Attach hls.js for other browsers
            var hls = new Hls();
            hls.loadSource(hlsPlaylist);
            hls.attachMedia(video);
        } else {
            console.error("This browser does not support MSE and cannot play HLS!");
        }
        </script>
    {% endif %}
    
    <h2>Latest Detections</h2>
    {% for date_label, day_items in grouped_detections.items() %}
    <h3>{{ date_label }}</h3>
    <ul>
        {% for detection in day_items %}
        <li>
            {{ detection.summary | safe }}
            {% if not detection.finalized %}
            <em>(Ongoing)</em>
            {% endif %}
            {# Check if there are images AND if the detection has a UUID #}
            {% if detection.image_filenames and detection.uuid %}
            <button
                class="view-images-btn"
                data-detection-uuid="{{ detection.uuid }}" {# <-- CHANGED: Use UUID #}
            >
                View Images
            </button>
            {% endif %}
        </li>
        {% endfor %}
    </ul>
    {% endfor %}

    <!-- Modal Structure -->
    <div id="image-modal">
    <span onclick="closeModal()">&times;</span>
    <div id="modal-content">
        <div id="loading-spinner"></div> <!-- Loading Spinner -->
        <img id="modal-image" src="">
        <div id="image-count"></div>
    </div>
    <!-- Existing left/right navigation buttons remain -->
    <button id="prev-button"></button>
    <button id="next-button"></button>
    <!-- New container for center buttons -->
    <div id="center-buttons">
        <button id="first-button">First</button>
        <button id="skip-back-button">«10</button>                                  
        <button id="swap-button">Swap Type</button>
        <button id="skip-forward-button">10»</button>                                  
        <button id="last-button">Last</button>
    </div>
    </div>

    <h2>Backend Logs</h2>
    <ul id="logs-list">
        {% for log in logs %}
            <li>{{ log }}</li>
        {% endfor %}
    </ul>

    <h2>Detection Graphs</h2>
    <form id="graph-form" action="{{ base_path }}/" method="get">
        <label for="hours">Select Time Range:</label>
        <select name="hours" id="hours">
            <option value="1">Last 1 Hour</option>
            <option value="3">Last 3 Hours</option>
            <option value="24">Last 24 Hours</option>
            <option value="168">Last Week</option>
            <option value="720">Last Month</option>
        </select>
        <input type="submit" value="View Graph">
    </form>

    <!-- Include the graph image if available -->
    {% if graph_image %}
        <div id="detection-graph">
            <h3>Detections Over Time</h3>
            <img src="data:image/png;base64,{{ graph_image }}">
        </div>
    {% else %}
        <div id="detection-graph">
            <h3>Detections Over Time</h3>
            <p style="color: red; font-weight: bold;">No detection data available for this time range.</p>
        </div>
    {% endif %}

      
<script>
        // Global variables
        const basePath = "{{ base_path }}"; // Define basePath early
        let modalOpen = false;
        let currentIndex = 0;
        let imageFilenames = [];
        let showFullFrame = true;

        // DOM Element references (grab them once for efficiency)
        const loadingSpinner = document.getElementById('loading-spinner');
        const modal = document.getElementById('image-modal');
        const modalContent = document.getElementById('modal-content');
        const modalImage = document.getElementById('modal-image');
        const swapButton = document.getElementById('swap-button');
        const imageCountElement = document.getElementById('image-count');
        const prevButton = document.getElementById('prev-button');
        const nextButton = document.getElementById('next-button');
        const firstButton = document.getElementById('first-button');
        const skipBackButton = document.getElementById('skip-back-button');
        const skipForwardButton = document.getElementById('skip-forward-button');
        const lastButton = document.getElementById('last-button');
        const logsListElement = document.getElementById('logs-list');
        const hostTimeElement = document.getElementById('host-time');
        const graphForm = document.getElementById('graph-form');
        const graphContainer = document.getElementById('detection-graph');
        const hoursSelect = document.getElementById('hours');
        const modalCloseButton = document.querySelector('#image-modal span'); // Get close button

        // === FUNCTION DEFINITIONS START ===

        // Function to clear excessive error messages in modal
        function clearErrorMessage() {
            const errorMessage = document.getElementById('error-message');
            if (errorMessage) {
                errorMessage.remove();
            }
        }

        // Function to display the modal with images
        function showImages(images) {
            console.log('showImages called with:', images);
            imageFilenames = images;
            currentIndex = 0;

            // *** CORRECTED LOGIC: Default to detection_area (showFullFrame = false) ***
            showFullFrame = false; // Default to showing detection area

            // Check the *first* image. If detection_area is MISSING, but full_frame EXISTS,
            // then switch the default to show full_frame instead.
            if (imageFilenames.length > 0) {
                const firstImageInfo = imageFilenames[0];
                if (!firstImageInfo.detection_area && firstImageInfo.full_frame) {
                    console.log("First image missing detection_area, defaulting to full_frame");
                    showFullFrame = true;
                } else if (!firstImageInfo.detection_area && !firstImageInfo.full_frame) {
                    console.warn("First image missing both detection_area and full_frame!");
                    // Keep showFullFrame as false, showImage will handle missing file
                }
            }
            // *** END OF CORRECTED LOGIC ***

            modal.style.display = 'flex';
            showImage(currentIndex); // Load the image based on the determined showFullFrame state
            modalOpen = true;
        }

        // Function to load and display a specific image in the modal
        function showImage(index) {
            if (!imageFilenames || index < 0 || index >= imageFilenames.length) {
                console.error('showImage: Invalid index or imageFilenames not set', index, imageFilenames);
                return;
            }

            const imageInfo = imageFilenames[index];
            console.log('Showing image at index:', index, 'Info:', imageInfo);
            let filename = null;

            // Determine which filename to use (full_frame or detection_area)
            if (showFullFrame && imageInfo.full_frame) {
                filename = imageInfo.full_frame;
            } else if (!showFullFrame && imageInfo.detection_area) {
                filename = imageInfo.detection_area;
            } else {
                // Fallback: use whichever is available, preferring full_frame
                filename = imageInfo.full_frame || imageInfo.detection_area;
                // Update showFullFrame state if we fell back
                if (filename === imageInfo.full_frame) showFullFrame = true;
                else if (filename === imageInfo.detection_area) showFullFrame = false;
            }

            if (!filename) {
                 console.error('No valid image filename found for index:', index, 'Image info:', imageInfo);
                 loadingSpinner.style.display = 'none'; // Hide spinner if no filename
                 modalImage.src = ''; // Clear image
                 modalImage.alt = 'Image not available.';
                 imageCountElement.textContent = `Image ${index + 1} of ${imageFilenames.length} (Not Available)`;
                 clearErrorMessage(); // Clear previous errors
                 return;
            }

            console.log('Loading image:', filename, 'showFullFrame:', showFullFrame);

            // Show loading spinner, clear previous errors/image
            loadingSpinner.style.display = 'block';
            modalImage.src = ''; // Clear previous image immediately
            modalImage.alt = 'Loading...';
            clearErrorMessage();

            // Update the image count display
            imageCountElement.textContent = `Image ${index + 1} of ${imageFilenames.length}`;

            // Construct URL, avoiding double slashes
            const imageURL = `${basePath.replace(/\/$/, '')}/api/detections/${encodeURI(filename)}`;
            modalImage.src = imageURL;
        }

        // Function to close the image modal
        function closeModal() {
            modal.style.display = 'none';
            modalOpen = false;
            modalImage.src = ''; // Clear image src
            imageFilenames = []; // Clear the stored filenames
            clearErrorMessage(); // Clear any lingering errors
        }

        // Function to fetch and update logs (if needed dynamically)
        function fetchLogs() {
            const logsURL = `${basePath.replace(/\/$/, '')}/api/logs`;
            fetch(logsURL)
                .then(response => response.json())
                .then(data => {
                    logsListElement.innerHTML = ''; // Clear existing logs
                    data.forEach(log => {
                        const li = document.createElement('li');
                        li.textContent = log;
                        logsListElement.appendChild(li);
                    });
                })
                .catch(error => console.error('Error fetching logs:', error));
        }

        // Function to fetch and update current host time
        function fetchCurrentTime() {
             const timeURL = `${basePath.replace(/\/$/, '')}/api/current_time`;
            fetch(timeURL)
                .then(response => {
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    return response.json();
                 })
                .then(data => {
                    if (hostTimeElement) {
                        hostTimeElement.textContent = data.current_time;
                    }
                })
                .catch(error => {
                    console.error('Error fetching current time:', error);
                     if (hostTimeElement) {
                        hostTimeElement.textContent = 'Error'; // Indicate error
                    }
                });
        }

        // === FUNCTION DEFINITIONS END ===


        // === EVENT LISTENERS START ===

        // --- Main click listener (delegated for 'View Images' buttons) ---
        document.addEventListener('click', function(e) {
            // --- 'View Images' button click ---
            if (e.target.matches('.view-images-btn')) {
                const button = e.target; // Keep reference to the button
                const detectionUUID = button.getAttribute('data-detection-uuid');
                if (!detectionUUID) {
                    console.error("Button is missing data-detection-uuid attribute");
                    alert("Could not load images: Missing detection identifier.");
                    return;
                }

                button.textContent = 'Loading...';
                button.disabled = true;

                console.log(`Fetching images for detection UUID: ${detectionUUID}`);

                // Construct URL carefully, avoiding double slashes
                const fetchURL = `${basePath.replace(/\/$/, '')}/api/detection_images/${detectionUUID}`;

                fetch(fetchURL)
                    .then(response => {
                        if (!response.ok) {
                            // Try to parse JSON error first, fallback to status text
                            return response.json()
                                .then(errData => { throw new Error(`HTTP ${response.status}: ${errData.error || 'Server error'}`); })
                                .catch(() => { throw new Error(`HTTP error ${response.status}`); }); // Fallback if error isn't JSON
                        }
                        return response.json();
                    })
                    .then(fetchedImageFilenames => {
                        if (fetchedImageFilenames && Array.isArray(fetchedImageFilenames)) {
                            if (fetchedImageFilenames.length > 0) {
                                showImages(fetchedImageFilenames); // Call showImages with the fetched list
                            } else {
                                console.warn(`No images found for detection ${detectionUUID}.`);
                                alert("No images are associated with this detection event.");
                            }
                        } else {
                            console.error("Received invalid data instead of image list:", fetchedImageFilenames);
                            throw new Error("Invalid data received from server.");
                        }
                    })
                    .catch(err => {
                        console.error("Error fetching or processing image list:", err);
                        alert(`Could not load images: ${err.message}.`);
                    })
                    .finally(() => {
                        // Reset button state only if it still exists in the DOM
                        if (document.body.contains(button)) {
                             button.textContent = 'View Images';
                             button.disabled = false;
                        }
                    });
            }
        });
        // --- END Main click listener ---

        // --- Modal Image Load/Error Handlers ---
         modalImage.onload = function() {
            loadingSpinner.style.display = 'none';
            clearErrorMessage();
         };

         modalImage.onerror = function() {
            // Check if src is not empty and not just the base URL to prevent false errors
            if (modalImage.src && !modalImage.src.endsWith(window.location.pathname)) {
                loadingSpinner.style.display = 'none';
                modalImage.alt = 'Failed to load image.';
                console.error('Failed to load image:', modalImage.src);

                let errorMessage = document.getElementById('error-message');
                if (!errorMessage) {
                    errorMessage = document.createElement('div');
                    errorMessage.id = 'error-message';
                    errorMessage.style.color = 'red'; // Make error obvious
                    errorMessage.style.marginTop = '10px';
                    errorMessage.style.padding = '5px';
                    errorMessage.style.backgroundColor = 'rgba(0,0,0,0.7)'; // Dark background for text
                    errorMessage.style.borderRadius = '3px';
                    errorMessage.style.fontWeight = 'bold';
                    // Insert after the image, before the count
                    modalContent.insertBefore(errorMessage, imageCountElement);
                }
                 errorMessage.textContent = 'Failed to load image.';
            } else {
                 // Ignore error if src was empty (e.g., on modal close/clear)
                 loadingSpinner.style.display = 'none';
                 modalImage.alt = ''; // Clear alt text
            }
         };

        // --- Modal Button Listeners ---
        swapButton.onclick = function() {
            showFullFrame = !showFullFrame;
            showImage(currentIndex); // Reload current index with new type
        };

        prevButton.onclick = function() {
            if (currentIndex > 0) {
                currentIndex--;
                showImage(currentIndex);
            }
        };

        nextButton.onclick = function() {
            if (currentIndex < imageFilenames.length - 1) {
                currentIndex++;
                showImage(currentIndex);
            }
        };

        firstButton.onclick = function() {
            if (imageFilenames.length > 0) {
                currentIndex = 0;
                showImage(currentIndex);
            }
        };

        skipBackButton.onclick = function() {
            if (imageFilenames.length > 0) {
                currentIndex = Math.max(currentIndex - 10, 0);
                showImage(currentIndex);
            }
        };

        skipForwardButton.onclick = function() {
            if (imageFilenames.length > 0) {
                currentIndex = Math.min(currentIndex + 10, imageFilenames.length - 1);
                showImage(currentIndex);
            }
        };

        lastButton.onclick = function() {
            if (imageFilenames.length > 0) {
                currentIndex = imageFilenames.length - 1;
                showImage(currentIndex);
            }
        };

        // Open image in new tab on click (ensure src is valid first)
         modalImage.onclick = function() {
             if (modalImage.src && !modalImage.src.endsWith(window.location.pathname)) {
                window.open(modalImage.src, '_blank');
             }
         };

        // Close modal using the 'X' button
        if (modalCloseButton) {
             modalCloseButton.onclick = closeModal; // Use the function directly
        }

        // Close modal when clicking on the background overlay
        modal.addEventListener('click', function(event) {
            if (event.target === modal) { // Check if the click was *directly* on the modal background
                closeModal();
            }
        });

        // --- Keyboard navigation for modal ---
        document.addEventListener('keydown', function(event) {
            if (!modalOpen) {
                return; // Only handle keys if modal is open
            }
            let handled = false;
            switch (event.key) {
                case 'ArrowLeft':
                    prevButton.onclick(); // Simulate button click
                    handled = true;
                    break;
                case 'ArrowRight':
                    nextButton.onclick(); // Simulate button click
                    handled = true;
                    break;
                case 'Escape':
                    closeModal();
                    handled = true;
                    break;
                case 'Home':
                     firstButton.onclick();
                     handled = true;
                     break;
                case 'End':
                     lastButton.onclick();
                     handled = true;
                     break;
                // Add PageUp/PageDown for skip?
                case 'PageDown':
                     skipForwardButton.onclick();
                     handled = true;
                     break;
                case 'PageUp':
                     skipBackButton.onclick();
                     handled = true;
                     break;
                 case ' ': // Spacebar to toggle type?
                     swapButton.onclick();
                     handled = true;
                     break;
            }
            if (handled) {
                 event.preventDefault(); // Prevent default browser action (like scrolling) for handled keys
            }
        });

        // --- Swipe gestures for modal (touch devices) ---
        let touchStartX = null;
        let touchStartY = null; // To distinguish scroll from swipe
        modal.addEventListener('touchstart', function(event) {
            if (event.touches.length === 1) { // Only care about single touch swipes
                touchStartX = event.touches[0].clientX; // Use clientX for horizontal position
                touchStartY = event.touches[0].clientY;
            } else {
                touchStartX = null;
                touchStartY = null;
            }
        }, { passive: true }); // Passive listener for starting touch

        modal.addEventListener('touchend', function(event) {
            if (touchStartX === null || touchStartY === null || event.changedTouches.length !== 1) {
                return; // Not a valid single touch swipe sequence
            }

            let touchEndX = event.changedTouches[0].clientX;
            let touchEndY = event.changedTouches[0].clientY;
            let diffX = touchStartX - touchEndX;
            let diffY = touchStartY - touchEndY;
            const swipeThreshold = 50; // Minimum horizontal movement for swipe
            const verticalThreshold = 75; // Max vertical movement allowed for horizontal swipe

            // Check if it's primarily a horizontal swipe and exceeds threshold
            if (Math.abs(diffX) > swipeThreshold && Math.abs(diffY) < verticalThreshold) {
                if (diffX > 0) { // Swiped left (finger moved left -> content moves left -> next image)
                    nextButton.onclick(); // Simulate next click
                } else { // Swiped right (finger moved right -> content moves right -> previous image)
                    prevButton.onclick(); // Simulate prev click
                }
                 // Optionally prevent default if the swipe was handled, though touchend is tricky
            }

            // Reset for next potential swipe
            touchStartX = null;
            touchStartY = null;
        }, false); // Not passive because we might *implicitly* be preventing scroll if swipe detected


        // --- Graph form submission ---
        graphForm.addEventListener('submit', function(event) {
            event.preventDefault(); // Prevent default page reload
            const hours = hoursSelect.value;
            const graphURL = `${basePath.replace(/\/$/, '')}/api/detection_graph/${hours}`;

            // Show loading state for graph
            graphContainer.innerHTML = '<h3>Detections Over Time</h3><p>Loading graph...</p>';

            fetch(graphURL)
                .then(response => {
                    if (!response.ok) {
                        return response.text().then(text => {
                             throw new Error(`HTTP ${response.status}: ${text || 'Failed to load graph'}`);
                        });
                    }
                    return response.text(); // Expecting HTML fragment with <img> tag or error <p>
                })
                .then(html => {
                    // Inject the fetched HTML (contains the img tag or error message)
                    graphContainer.innerHTML = '<h3>Detections Over Time</h3>' + html;
                })
                .catch(error => {
                    console.error('Error fetching graph:', error);
                    graphContainer.innerHTML = `<h3>Detections Over Time</h3><p style="color: red; font-weight: bold;">Failed to load detection graph: ${error.message}</p>`;
                });
        });

        // === EVENT LISTENERS END ===


        // === INTERVAL TIMERS ===
        // Fetch current time every second
        const timeIntervalId = setInterval(fetchCurrentTime, 1000);

        // Optional: Fetch logs periodically (Example)
        // const logIntervalId = setInterval(fetchLogs, 30000); // e.g., every 30 seconds

        // === INITIAL ACTIONS ===
        // Fetch time immediately on page load
        fetchCurrentTime();

        // Optional: Fetch logs immediately on page load (if needed beyond server render)
        // fetchLogs();

        // Optional: Trigger graph load for default selection on page load
        // graphForm.dispatchEvent(new Event('submit'));


        // === CLEANUP (Optional) ===
        // Example: Clear intervals if the page were being unloaded in a SPA context
        // window.addEventListener('unload', () => {
        //    clearInterval(timeIntervalId);
        //    clearInterval(logIntervalId);
        // });

    </script>
                                      
    <!-- Horizontal Rule and Footer with Version Number -->
    <hr>
    <footer>
        <p>version: {{ version }}</p>
    </footer>
                                  
</body>
</html>
    ''', detections=detections, logs=logs, graph_image=graph_image, base_path=base_path, version=version_number, preview_method=preview_method, grouped_detections=grouped_detections)

@app.route('/api/detection_graph/<int:hours>')
def detection_graph_route(hours):
    """Route to serve the detection graph for the specified time range."""
    detection_log_path = app.config.get('detection_log_path')  # Safely get the path
    if not detection_log_path:
        logger.error("Detection log path is not set.")
        return '<p style="color: red; font-weight: bold;">Internal Server Error.</p>', 500

    image_base64 = generate_detection_graph(hours, detection_log_path)
    if image_base64 is None:
        logger.info("User tried to request a detection data graph, but it was not available.")
        return '<p style="color: red; font-weight: bold;">No detection data available for this time range.</p>'

    return f'<img src="data:image/png;base64,{image_base64}">'

def new_detection(detection):
    """Add a new detection to the list."""
    with app.config['detections_lock']:
        app.config['detections_list'].append(detection)

def new_log(log_message):
    """Add a new log to the list."""
    with app.config['logs_lock']:
        app.config['logs_list'].append(log_message)

# Example usage: Simulate adding detections and logs (Replace with actual detection logic)
def simulate_detection_and_logging():
    """Simulate adding detections and logs for demonstration purposes."""
    frame_count = 0
    while True:
        time.sleep(2)  # Simulate detection every 2 seconds
        frame_count += 1
        detection = {
            'frame_count': frame_count,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'coordinates': f'({100 + frame_count}, {200 + frame_count})',
            'confidence': 0.95
        }
        new_detection(detection)
        new_log(f'Detection {frame_count} added.')

# Start the simulation in a separate thread (Remove or replace in production)
# detection_thread = threading.Thread(target=simulate_detection_and_logging, daemon=True)
# detection_thread.start()
# Removed simulation code for production use

# new candidate selection
def get_base_save_dir(config):
    # This function is now just a wrapper around the candidate logic,
    # returning the first valid one for when the server is run standalone.
    candidates = get_save_dir_candidates(config)
    
    for directory in candidates:
        if not directory: continue
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            if os.path.isdir(directory) and os.access(directory, os.W_OK):
                logger.info(f"Web server standalone mode determined valid save dir: {directory}")
                return directory
        except Exception:
            continue
            
    logger.error("No writable save directory available for web server (standalone).")
    raise RuntimeError("No writable save directory available.")

# def get_base_save_dir(config):
#     base_save_dir = None

#     # Read configurations
#     USE_ENV_SAVE_DIR = config.getboolean('detection', 'use_env_save_dir', fallback=False)
#     ENV_SAVE_DIR_VAR = config.get('detection', 'env_save_dir_var', fallback='YOLO_SAVE_DIR')
#     logger.info(f"USE_ENV_SAVE_DIR: {USE_ENV_SAVE_DIR}")
#     logger.info(f"ENV_SAVE_DIR_VAR: {ENV_SAVE_DIR_VAR}")

#     # Check environment variable
#     if USE_ENV_SAVE_DIR:
#         env_dir = os.getenv(ENV_SAVE_DIR_VAR)
#         logger.info(f"Environment variable {ENV_SAVE_DIR_VAR} value: {env_dir}")
#         if env_dir and os.path.exists(env_dir) and os.access(env_dir, os.W_OK):
#             logger.info(f"Using environment-specified save directory: {env_dir}")
#             base_save_dir = env_dir
#         else:
#             logger.warning(f"Environment variable {ENV_SAVE_DIR_VAR} is set but the directory does not exist or is not writable. Checked path: {env_dir}")

#     # Fallback to config value
#     if not base_save_dir:
#         SAVE_DIR_BASE = config.get('detection', 'save_dir_base', fallback='./yolo_detections')
#         logger.info(f"Attempting to use save_dir_base from config: {SAVE_DIR_BASE}")
#         if os.path.exists(SAVE_DIR_BASE) and os.access(SAVE_DIR_BASE, os.W_OK):
#             logger.info(f"Using save_dir_base from config: {SAVE_DIR_BASE}")
#             base_save_dir = SAVE_DIR_BASE
#         else:
#             logger.warning(f"save_dir_base {SAVE_DIR_BASE} does not exist or is not writable. Attempting to create it.")
#             try:
#                 os.makedirs(SAVE_DIR_BASE, exist_ok=True)
#                 if os.access(SAVE_DIR_BASE, os.W_OK):
#                     logger.info(f"Created and using save_dir_base: {SAVE_DIR_BASE}")
#                     base_save_dir = SAVE_DIR_BASE
#                 else:
#                     logger.warning(f"save_dir_base {SAVE_DIR_BASE} is not writable after creation.")
#             except Exception as e:
#                 logger.error(f"Failed to create save_dir_base: {SAVE_DIR_BASE}. Error: {e}")

#     if not base_save_dir:
#         logger.error("No writable save directory available. Exiting.")
#         raise RuntimeError("No writable save directory available.")

#     logger.info(f"Final base_save_dir: {base_save_dir}")
#     return base_save_dir

if __name__ == '__main__':
    # Load configurations
    config = load_config()

    # Extract necessary configurations
    host = config.get('webserver', 'webserver_host', fallback='0.0.0.0')
    port = config.getint('webserver', 'webserver_port', fallback=5000)

    # Initialize shared resources
    detections_list = deque(maxlen=100)
    logs_list = deque(maxlen=100)
    detections_lock = threading.Lock()
    logs_lock = threading.Lock()

    # Define detection_log_path
    detection_log_path = detection_log_file  # This line is added

    # Start the web server without passing save_dir_base
    start_web_server(
        host=host,
        port=port,
        detection_log_path=detection_log_path,
        detections_list=detections_list,
        logs_list=logs_list,
        detections_lock=detections_lock,
        logs_lock=logs_lock,
        config=config
    )
