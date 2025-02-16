
# web_server.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Web server module for real-time YOLOv8 detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Version number
import version  # Import the version module
version_number = version.version_number

import sys
import os
import signal
import shutil

from collections import deque
from datetime import datetime
import threading
import time
from flask import Flask, Response, render_template_string, stream_with_context, request, jsonify
from flask import send_from_directory
from flask import send_file
from werkzeug.utils import safe_join
import cv2
import logging
from web_graph import generate_detection_graph
import configparser
import json

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
    config = configparser.ConfigParser()
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
WEBUI_COOLDOWN_AGGREGATION = config.getint('webui', 'webui_cooldown_aggregation', fallback=30)
WEBUI_BOLD_THRESHOLD = config.getint('webui', 'webui_bold_threshold', fallback=10)
# Read check_interval from config.ini with a fallback to 10
interval_checks = config.getboolean('webserver', 'interval_checks', fallback=True)
check_interval = config.getint('webserver', 'check_interval', fallback=10)
# Persistent aggregated detections
ENABLE_PERSISTENT_AGGREGATED_DETECTIONS = config.getboolean('aggregation', 'enable_persistent_aggregated_detections', fallback=False)
AGGREGATED_DETECTIONS_FILE = config.get('aggregation', 'aggregated_detections_file', fallback='./logs/aggregated_detections.json')

if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
    logger.info(f"Persistent aggregated detections enabled. Logging to file: {AGGREGATED_DETECTIONS_FILE}")

# Gglobal variables for aggregation
# aggregated_detections_list = deque(maxlen=100)  # Adjust maxlen as needed

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
    logger.info(f"Check Interval: {config.getint('webserver', 'check_interval', fallback=10)} seconds")
    logger.info(f"Web UI Cooldown Aggregation: {config.getint('webui', 'webui_cooldown_aggregation', fallback=30)} seconds")
    logger.info(f"Web UI Bold Threshold: {config.getint('webui', 'webui_bold_threshold', fallback=10)}")
    logger.info(f"Persistent Aggregated Detections Enabled: {ENABLE_PERSISTENT_AGGREGATED_DETECTIONS}")
    if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
        logger.info(f"Aggregated Detections File: {AGGREGATED_DETECTIONS_FILE}")    
    logger.info(f"SAVE_DIR_BASE is set to: {app.config['SAVE_DIR_BASE']}")
    logger.info("======================================================")

    # app.run(host=host, port=port, threaded=True)
    app.run(host=host, port=port, threaded=True, use_reloader=False)    

def set_output_frame(frame):
    """Updates the global output frame to be served to clients."""
    global output_frame
    with frame_lock:
        output_frame = frame.copy()

def generate_frames():
    """Generator function that yields frames in byte format for streaming."""
    global output_frame
    max_fps = app.config.get('webserver_max_fps', 10)  # Default to 10 FPS if not set
    frame_interval = 1.0 / max_fps
    last_frame_time = time.time()
    while True:
        with frame_lock:
            if output_frame is None:
                frame_copy = None
            else:
                # Make a copy of the frame
                frame_copy = output_frame.copy()
        
        if frame_copy is None:
            # Sleep briefly to prevent 100% CPU utilization
            time.sleep(0.01)
            continue

        # Limit the frame rate
        current_time = time.time()
        elapsed_time = current_time - last_frame_time
        if elapsed_time < frame_interval:
            time.sleep(frame_interval - elapsed_time)
        last_frame_time = time.time()

        # Encode the frame in JPEG format outside the lock
        ret, jpeg = cv2.imencode('.jpg', frame_copy)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()

        # Yield the output frame in byte format
        try:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            # Client disconnected
            break
        except Exception as e:
            logger.error(f"Error in streaming frames: {e}")
            break

# aggergation for the detections for webUI
# At the beginning of the file
from collections import defaultdict

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
                        # Start a new aggregation
                        current_aggregation = {
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

                if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
                    save_aggregated_detections()

            # Check if cooldown period has passed since the last detection
            if current_aggregation and last_detection_time and (time.time() - last_detection_time) >= cooldown:
                # Mark current aggregation as finalized
                current_aggregation['finalized'] = True
                current_aggregation = None
                last_detection_time = None

                if ENABLE_PERSISTENT_AGGREGATED_DETECTIONS:
                    save_aggregated_detections()

        except Exception as e:
            logger.error(f"Error in aggregation_thread_function: {e}", exc_info=True)

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
    save_dir_base = app.config.get('SAVE_DIR_BASE', './yolo_detections')
    logger.info(f"SAVE_DIR_BASE: {save_dir_base}")
    logger.info(f"Requested filename: {filename}")
    # Use os.path.join and os.path.abspath
    filepath = os.path.abspath(os.path.join(save_dir_base, filename))
    logger.info(f"Computed filepath: {filepath}")

    # Security check to prevent directory traversal
    if not save_dir_base:
        logger.error("save_dir_base is None. Cannot serve image.")
        return "Internal Server Error", 500
    if not filepath.startswith(os.path.abspath(save_dir_base)):
        logger.error("Attempted directory traversal attack")
        return "Forbidden", 403
    if os.path.isfile(filepath):
        return send_file(filepath)
    else:
        logger.error(f"File not found: {filepath}")
        return "File not found", 404

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

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(stream_with_context(generate_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    })

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
            user-drag: none;
            user-select: none;                                  
            cursor: zoom-in;
        }

        #image-modal img {
            max-width: 100%;
            max-height: 70%;
            margin-bottom: 10px;
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
</head>
<body>
    <script>
    const basePath = "{{ base_path }}";
    </script>                                  

    <!-- Time Display Container -->
    <div id="time-container">
        <strong>Current Host Time:</strong> <span id="host-time">Loading...</span>
    </div>

    <h1>Real-time Human Detection</h1>
    <img src="{{ base_path }}{{ url_for('video_feed') }}" width="100%">

    <h2>Latest Detections</h2>
    <ul id="detections-list">
        {% for detection in detections %}
            <li>
                {{ detection.summary | safe }}
                {% if not detection.finalized %}
                    <em>(Ongoing)</em>
                {% endif %}
                {% if detection.image_filenames %}
                    - <a href="#" onclick="showImages({{ detection.image_filenames | tojson }}); return false;">View Images</a>
                {% endif %}
            </li>
        {% endfor %}
    </ul>

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
        let modalOpen = false;
        let currentIndex = 0;
        let imageFilenames = [];
        let showFullFrame = true;
                          
        const loadingSpinner = document.getElementById('loading-spinner');
        const modal = document.getElementById('image-modal');
        const modalContent = document.getElementById('modal-content');
        const modalImage = document.getElementById('modal-image');
        const swapButton = document.getElementById('swap-button');
        const imageCountElement = document.getElementById('image-count');
        const prevButton = document.getElementById('prev-button');
        const nextButton = document.getElementById('next-button');

        // Function to clear excessive error messages
        function clearErrorMessage() {
            const errorMessage = document.getElementById('error-message');
            if (errorMessage) {
                errorMessage.remove();
            }
        }

        function showImages(images) {
            console.log('showImages called with:', images);
            imageFilenames = images;
            currentIndex = 0;
            showFullFrame = false; // Start with detection area images

            modal.style.display = 'flex';
            showImage(currentIndex);
            modalOpen = true;
        }

        function showImage(index) {
            const imageInfo = imageFilenames[index];
            console.log('imageInfo:', imageInfo);
            let filename = null;

            if (showFullFrame && imageInfo.full_frame) {
                filename = imageInfo.full_frame;
            } else if (!showFullFrame && imageInfo.detection_area) {
                filename = imageInfo.detection_area;
            } else if (imageInfo.full_frame || imageInfo.detection_area) {
                // Fallback to whichever image is available
                filename = imageInfo.full_frame || imageInfo.detection_area;
            } else {
                console.error('No valid image filename found for:', imageInfo);
                return; // Exit the function if no valid filename
            }

            console.log('Loading image:', filename);

            // Show loading spinner
            loadingSpinner.style.display = 'block';

            // Clear any existing error messages
            clearErrorMessage();

            // Update the image count display
            imageCountElement.textContent = `Image ${index + 1} of ${imageFilenames.length}`;

            // Set the event handlers
            modalImage.onload = function() {
                loadingSpinner.style.display = 'none';
                clearErrorMessage();
            };

            // Open the detection image in a new window when clicked on
            modalImage.onclick = function() {
                window.open(modalImage.src, '_blank');
            };
                                                                    
            modalImage.onerror = function() {
                // Check if src is not empty to prevent false errors
                if (modalImage.src) {
                    loadingSpinner.style.display = 'none';
                    modalImage.alt = 'Failed to load image.';

                    console.error('Failed to load image:', modalImage.src);

                    let errorMessage = document.getElementById('error-message');
                    if (!errorMessage) {
                        errorMessage = document.createElement('div');
                        errorMessage.id = 'error-message';
                        errorMessage.style.color = 'white';
                        errorMessage.style.marginTop = '10px';
                        errorMessage.textContent = 'Failed to load image. Please try again later.';
                        modalContent.appendChild(errorMessage);
                    }
                }
            };

            // Set the image source to start loading
            modalImage.src = `${basePath}/api/detections/${encodeURI(filename)}`;
        }

        modalImage.onload = function() {
            loadingSpinner.style.display = 'none';
            clearErrorMessage();
        };

        modalImage.onerror = function() {
            loadingSpinner.style.display = 'none';
            modalImage.alt = 'Failed to load image.';

            console.error('Failed to load image:', modalImage.src);

            let errorMessage = document.getElementById('error-message');
            if (!errorMessage) {
                errorMessage = document.createElement('div');
                errorMessage.id = 'error-message';
                errorMessage.style.color = 'white';
                errorMessage.style.marginTop = '10px';
                errorMessage.textContent = 'Failed to load image. Please try again later.';
                modalContent.appendChild(errorMessage);
            }
        };

        swapButton.onclick = function() {
            console.log("Swap button clicked");
            showFullFrame = !showFullFrame;
            showImage(currentIndex);
        };

        prevButton.onclick = function() {
            console.log("Previous button clicked");
            if (currentIndex > 0) {
                currentIndex--;
                showImage(currentIndex);
            }
        };

        nextButton.onclick = function() {
            console.log("Next button clicked");
            if (currentIndex < imageFilenames.length - 1) {
                currentIndex++;
                showImage(currentIndex);
            }
        };

        document.getElementById('first-button').onclick = function() {
        console.log("First image button clicked");
        if (imageFilenames && imageFilenames.length > 0) {
            currentIndex = 0;
            showImage(currentIndex);
        }
        };

        document.getElementById('skip-back-button').onclick = function() {
        console.log("Skip back 10 button clicked");
        if (imageFilenames && imageFilenames.length > 0) {
            // Subtract 10, but don’t go below 0
            currentIndex = Math.max(currentIndex - 10, 0);
            showImage(currentIndex);
        }
        };

        document.getElementById('skip-forward-button').onclick = function() {
        console.log("Skip forward 10 button clicked");
        if (imageFilenames && imageFilenames.length > 0) {
            // Add 10, but don’t go past the last image
            currentIndex = Math.min(currentIndex + 10, imageFilenames.length - 1);
            showImage(currentIndex);
        }
        };
                                  
        document.getElementById('last-button').onclick = function() {
        console.log("Last image button clicked");
        if (imageFilenames && imageFilenames.length > 0) {
            currentIndex = imageFilenames.length - 1;
            showImage(currentIndex);
        }
        };

        function closeModal() {
            modal.style.display = 'none';
            modalOpen = false;
        }

        // Close modal when clicking outside modal content
        modal.addEventListener('click', function(event) {
            if (event.target === modal) {
                closeModal();
            }
        });

        // Attach the keydown event listener once
        document.addEventListener('keydown', function(event) {
            if (!modalOpen) {
                return;
            }
            if (event.keyCode === 37) { // Left arrow
                if (currentIndex > 0) {
                    currentIndex--;
                    showImage(currentIndex);
                }
            } else if (event.keyCode === 39) { // Right arrow
                if (currentIndex < imageFilenames.length - 1) {
                    currentIndex++;
                    showImage(currentIndex);
                }
            } else if (event.keyCode === 27) { // Escape key
                closeModal();
            }
        });

        // Implement swipe gestures for mobile devices
        let touchStartX = null;

        modal.addEventListener('touchstart', function(event) {
            touchStartX = event.changedTouches[0].screenX;
        }, false);

        modal.addEventListener('touchend', function(event) {
            if (touchStartX === null) {
                return;
            }

            let touchEndX = event.changedTouches[0].screenX;
            let diffX = touchStartX - touchEndX;

            if (Math.abs(diffX) > 30) { // Swipe threshold
                if (diffX > 0) {
                    // Swipe left - Next image
                    if (currentIndex < imageFilenames.length - 1) {
                        currentIndex++;
                        showImage(currentIndex);
                    }
                } else {
                    // Swipe right - Previous image
                    if (currentIndex > 0) {
                        currentIndex--;
                        showImage(currentIndex);
                    }
                }
            }

            touchStartX = null;
        }, false);

        // Function to fetch and update detections
        function fetchDetections() {
            fetch(`${basePath}/api/detections`)
                .then(response => response.json())
                .then(data => {
                    const detectionsList = document.getElementById('detections-list');
                    detectionsList.innerHTML = ''; // Clear existing detections
                    data.forEach(detection => {
                        const li = document.createElement('li');
                        li.innerHTML = detection.summary; // Use innerHTML to render HTML content

                        if (detection.image_filenames && detection.image_filenames.length > 0) {
                            // Add the "View Images" link
                            const linkText = document.createTextNode(' - ');
                            li.appendChild(linkText);

                            const link = document.createElement('a');
                            link.href = '#';
                            link.textContent = 'View Images';
                            link.onclick = function() {
                                showImages(detection.image_filenames);
                                return false;
                            };
                            li.appendChild(link);
                        }

                        detectionsList.appendChild(li);
                    });
                })
                .catch(error => console.error('Error fetching detections:', error));
        }

        // Function to fetch and update logs
        function fetchLogs() {
            fetch(`${basePath}/api/logs`)
                .then(response => response.json())
                .then(data => {
                    const logsList = document.getElementById('logs-list');
                    logsList.innerHTML = ''; // Clear existing logs
                    data.forEach(log => {
                        const li = document.createElement('li');
                        li.textContent = log;
                        logsList.appendChild(li);
                    });
                })
                .catch(error => console.error('Error fetching logs:', error));
        }

        // Function to fetch and update current host time
        function fetchCurrentTime() {
            fetch(`${basePath}/api/current_time`)
                .then(response => response.json())
                .then(data => {
                    const hostTimeElement = document.getElementById('host-time');
                    hostTimeElement.textContent = data.current_time;
                })
                .catch(error => console.error('Error fetching current time:', error));
        }

        // Set intervals to fetch data every 5 seconds
        setInterval(() => {
            fetchDetections();
            fetchLogs();
        }, 1000); // 5000 milliseconds = 5 seconds

        // Also fetch current time every second for real-time update
        setInterval(() => {
            fetchCurrentTime();
        }, 1000); // 1000 milliseconds = 1 second

        // Initial fetch when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            fetchDetections();
            fetchLogs();
            fetchCurrentTime(); // Initial fetch
        });

        // Handle form submission via AJAX to avoid page reload
        document.getElementById('graph-form').addEventListener('submit', function(event) {
            event.preventDefault(); // Prevent the default form submission
            const hours = document.getElementById('hours').value;
            // Fetch the graph image via AJAX
            fetch(`${basePath}/api/detection_graph/${hours}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.text();
                })
                .then(html => {
                    const graphContainer = document.getElementById('detection-graph');
                    graphContainer.innerHTML = '<h3>Detections Over Time</h3>' + html;
                })
                .catch(error => {
                    console.error('Error fetching graph:', error);
                    // Display the error message in the graph container
                    const graphContainer = document.getElementById('detection-graph');
                    graphContainer.innerHTML = '<h3>Detections Over Time</h3><p style="color: red; font-weight: bold;">Failed to load detection graph.</p>';
                });
        });
    </script>
                                  
    <!-- Horizontal Rule and Footer with Version Number -->
    <hr>
    <footer>
        <p>version: {{ version }}</p>
    </footer>
                                  
</body>
</html>
    ''', detections=detections, logs=logs, graph_image=graph_image, base_path=base_path, version=version_number)

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

def get_base_save_dir(config):
    base_save_dir = None

    # Read configurations
    USE_ENV_SAVE_DIR = config.getboolean('detection', 'use_env_save_dir', fallback=False)
    ENV_SAVE_DIR_VAR = config.get('detection', 'env_save_dir_var', fallback='YOLO_SAVE_DIR')
    logger.info(f"USE_ENV_SAVE_DIR: {USE_ENV_SAVE_DIR}")
    logger.info(f"ENV_SAVE_DIR_VAR: {ENV_SAVE_DIR_VAR}")

    # Check environment variable
    if USE_ENV_SAVE_DIR:
        env_dir = os.getenv(ENV_SAVE_DIR_VAR)
        logger.info(f"Environment variable {ENV_SAVE_DIR_VAR} value: {env_dir}")
        if env_dir and os.path.exists(env_dir) and os.access(env_dir, os.W_OK):
            logger.info(f"Using environment-specified save directory: {env_dir}")
            base_save_dir = env_dir
        else:
            logger.warning(f"Environment variable {ENV_SAVE_DIR_VAR} is set but the directory does not exist or is not writable. Checked path: {env_dir}")

    # Fallback to config value
    if not base_save_dir:
        SAVE_DIR_BASE = config.get('detection', 'save_dir_base', fallback='./yolo_detections')
        logger.info(f"Attempting to use save_dir_base from config: {SAVE_DIR_BASE}")
        if os.path.exists(SAVE_DIR_BASE) and os.access(SAVE_DIR_BASE, os.W_OK):
            logger.info(f"Using save_dir_base from config: {SAVE_DIR_BASE}")
            base_save_dir = SAVE_DIR_BASE
        else:
            logger.warning(f"save_dir_base {SAVE_DIR_BASE} does not exist or is not writable. Attempting to create it.")
            try:
                os.makedirs(SAVE_DIR_BASE, exist_ok=True)
                if os.access(SAVE_DIR_BASE, os.W_OK):
                    logger.info(f"Created and using save_dir_base: {SAVE_DIR_BASE}")
                    base_save_dir = SAVE_DIR_BASE
                else:
                    logger.warning(f"save_dir_base {SAVE_DIR_BASE} is not writable after creation.")
            except Exception as e:
                logger.error(f"Failed to create save_dir_base: {SAVE_DIR_BASE}. Error: {e}")

    if not base_save_dir:
        logger.error("No writable save directory available. Exiting.")
        raise RuntimeError("No writable save directory available.")

    logger.info(f"Final base_save_dir: {base_save_dir}")
    return base_save_dir

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
