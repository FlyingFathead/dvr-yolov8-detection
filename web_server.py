
# web_server.py
# (Updated Oct 11, 2024)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Web server module for real-time YOLOv8 detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import os
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

# Configure logging for the web server
logger = logging.getLogger('web_server')
logger.setLevel(logging.INFO)
# Prevent messages from propagating to the root logger
logger.propagate = False

app = Flask(__name__)
# Flask proxy fix
# from werkzeug.middleware.proxy_fix import ProxyFix
# app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
# app.config['APPLICATION_ROOT'] = '/'

# Global variables to hold the output frame and a lock for thread safety
output_frame = None
frame_lock = threading.Lock()

# New global variables for aggregation
aggregated_detections_list = deque(maxlen=100)  # Adjust maxlen as needed
aggregated_lock = threading.Lock()

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

# // client tracking
# Keep track of connected clients
connected_clients = {}

# Initialize the lock for connected_clients
connected_clients_lock = threading.Lock()

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

    # Initialize SAVE_DIR_BASE within the web server process
    SAVE_DIR_BASE = get_base_save_dir(config)
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
    last_detection_time = None
    current_aggregation = None  # To track ongoing aggregation

    while True:
        time.sleep(1)  # Check every second

        with detections_lock:
            if detections_list:
                # There are new detections
                for detection in detections_list:
                    timestamp = datetime.strptime(detection['timestamp'], '%Y-%m-%d %H:%M:%S')
                    confidence = detection['confidence']
                    image_filenames = detection.get('image_filenames', [])

                    if current_aggregation is None:
                        # Start a new aggregation
                        current_aggregation = {
                            'count': 1,
                            'first_timestamp': timestamp,
                            'latest_timestamp': timestamp,
                            'lowest_confidence': confidence,
                            'highest_confidence': confidence,
                            'image_filenames': image_filenames.copy()
                        }
                        # Create the summary string
                        summary = f"👀 Human detected {current_aggregation['count']} times within {cooldown} seconds! " \
                                  f"First seen: {current_aggregation['first_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}, " \
                                  f"Latest: {current_aggregation['latest_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}, " \
                                  f"Lowest confidence: {current_aggregation['lowest_confidence']:.2f}, " \
                                  f"Highest confidence: {current_aggregation['highest_confidence']:.2f}"
                        with aggregated_lock:
                            aggregated_detections_list.appendleft({
                                'summary': summary,
                                'image_filenames': current_aggregation['image_filenames']
                            })
                    else:
                        # Update the ongoing aggregation
                        current_aggregation['count'] += 1
                        if timestamp < current_aggregation['first_timestamp']:
                            current_aggregation['first_timestamp'] = timestamp
                        if timestamp > current_aggregation['latest_timestamp']:
                            current_aggregation['latest_timestamp'] = timestamp
                        if confidence < current_aggregation['lowest_confidence']:
                            current_aggregation['lowest_confidence'] = confidence
                        if confidence > current_aggregation['highest_confidence']:
                            current_aggregation['highest_confidence'] = confidence
                        current_aggregation['image_filenames'].extend(image_filenames)

                        # Determine if count meets or exceeds the bold_threshold
                        if current_aggregation['count'] >= bold_threshold:
                            count_display = f"<strong>{current_aggregation['count']}</strong>"
                        else:
                            count_display = f"{current_aggregation['count']}"

                        # Update the summary string with bolded count if applicable
                        summary = f"👀 Human detected {count_display} times within {cooldown} seconds! " \
                                  f"First seen: {current_aggregation['first_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}, " \
                                  f"Latest: {current_aggregation['latest_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}, " \
                                  f"Lowest confidence: {current_aggregation['lowest_confidence']:.2f}, " \
                                  f"Highest confidence: {current_aggregation['highest_confidence']:.2f}"
                        with aggregated_lock:
                            if len(aggregated_detections_list) > 0:
                                aggregated_detections_list[0]['summary'] = summary
                                aggregated_detections_list[0]['image_filenames'] = current_aggregation['image_filenames']
                            else:
                                aggregated_detections_list.appendleft({
                                    'summary': summary,
                                    'image_filenames': current_aggregation['image_filenames']
                                })

                # Update the last_detection_time to now
                last_detection_time = time.time()

                # Clear the detections_list after processing
                detections_list.clear()

        # Check if cooldown period has passed since the last detection
        if current_aggregation and last_detection_time:
            if (time.time() - last_detection_time) >= cooldown:
                # Finalize the current aggregation
                current_aggregation = None
                last_detection_time = None

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

@app.route('/detections/<path:filename>')
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
    with aggregated_lock:
        aggregated_detections = list(aggregated_detections_list)
    return jsonify(aggregated_detections)

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
        #image-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }

        #image-modal img {
            max-width: 90%;
            max-height: 90%;
        }

        #image-modal span {
            position: absolute;
            top: 10px;
            right: 20px;
            color: white;
            font-size: 30px;
            cursor: pointer;
        }                                  
    </style>
</head>
<body>
    <!-- Time Display Container -->
    <div id="time-container">
        <strong>Current Host Time:</strong> <span id="host-time">Loading...</span>
    </div>

    <h1>Real-time Human Detection</h1>
    <img src="{{ url_for('video_feed') }}" width="100%">

    <h2>Latest Detections</h2>
    <ul id="detections-list">
        {% for detection in detections %}
            <li>
                {{ detection.summary | safe }}
                {% if detection.image_filenames %}
                    - <a href="#" onclick="showImages({{ detection.image_filenames | tojson }}); return false;">View Images</a>
                {% endif %}
            </li>
        {% endfor %}
    </ul>

    <!-- Modal Structure -->
    <div id="image-modal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; 
        background-color:rgba(0,0,0,0.8); z-index:1000; align-items:center; justify-content:center;">
        <span style="position:absolute; top:10px; right:20px; color:white; font-size:30px; cursor:pointer;" onclick="closeModal()">&times;</span>
        <img id="modal-image" src="" style="max-width:90%; max-height:90%;">
    </div>
                                  
    <h2>Backend Logs</h2>
    <ul id="logs-list">
        {% for log in logs %}
            <li>{{ log }}</li>
        {% endfor %}
    </ul>
                  
    <h2>Detection Graphs</h2>
    <form id="graph-form" action="/" method="get">
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
        function showImages(imageFilenames) {
            let currentIndex = 0;
            const modal = document.getElementById('image-modal');
            const modalImage = document.getElementById('modal-image');

            function showImage(index) {
                const filename = imageFilenames[index];
                modalImage.src = '/detections/' + encodeURIComponent(filename);
            }

            modal.style.display = 'flex';
            showImage(currentIndex);

            // Navigate images with arrow keys
            document.onkeydown = function(event) {
                if (event.keyCode == 37) { // Left arrow
                    if (currentIndex > 0) {
                        currentIndex--;
                        showImage(currentIndex);
                    }
                } else if (event.keyCode == 39) { // Right arrow
                    if (currentIndex < imageFilenames.length - 1) {
                        currentIndex++;
                        showImage(currentIndex);
                    }
                } else if (event.keyCode == 27) { // Escape key
                    closeModal();
                }
            }
        }

        function closeModal() {
            const modal = document.getElementById('image-modal');
            modal.style.display = 'none';
            document.onkeydown = null;
        }


        // Function to fetch and update detections
        function fetchDetections() {
            fetch('/api/detections')
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
            fetch('/api/logs')
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
            fetch('/api/current_time')
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
            // fetchCurrentTime(); // Removed from here
        }, 5000); // 5000 milliseconds = 5 seconds

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
            fetch(`/detection_graph/${hours}`)
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
</body>
</html>
    ''', detections=detections, logs=logs, graph_image=graph_image)

@app.route('/detection_graph/<int:hours>')
def detection_graph_route(hours):
    """Route to serve the detection graph for the specified time range."""
    detection_log_path = app.config['detection_log_path']
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
