
# web_server.py
# (Updated Oct 11, 2024)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Web server module for real-time YOLOv8 detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from collections import deque
from datetime import datetime
import threading
import time
from flask import Flask, Response, render_template_string, stream_with_context, request, jsonify
import cv2
import logging
from web_graph import generate_detection_graph
import configparser

# Configure logging for the web server
logger = logging.getLogger('web_server')
logger.setLevel(logging.INFO)

# Prevent messages from propagating to the root logger
logger.propagate = False  # Add this line

app = Flask(__name__)

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

# Extract configurations with fallbacks
ENABLE_WEBSERVER = config.getboolean('webserver', 'enable_webserver', fallback=True)
WEBSERVER_HOST = config.get('webserver', 'webserver_host', fallback='0.0.0.0')
WEBSERVER_PORT = config.getint('webserver', 'webserver_port', fallback=5000)
WEBSERVER_MAX_FPS = config.getint('webserver', 'webserver_max_fps', fallback=10)
WEBUI_COOLDOWN_AGGREGATION = config.getint('webui', 'webui_cooldown_aggregation', fallback=30)
WEBUI_BOLD_THRESHOLD = config.getint('webui', 'webui_bold_threshold', fallback=10)

# Configure logging for the web server
logger = logging.getLogger('web_server')
logger.setLevel(logging.INFO)

# # Console handler
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

# Log the active configurations on startup
logger.info("======================================================")
logger.info("Web Server Configuration:")
logger.info(f"Enable Web Server: {ENABLE_WEBSERVER}")
logger.info(f"Web Server Host: {WEBSERVER_HOST}")
logger.info(f"Web Server Port: {WEBSERVER_PORT}")
logger.info(f"Web Server Max FPS: {WEBSERVER_MAX_FPS}")
logger.info(f"Web UI Cooldown Aggregation: {WEBUI_COOLDOWN_AGGREGATION} seconds")
logger.info(f"Web UI Bold Threshold: {WEBUI_BOLD_THRESHOLD}")
logger.info("======================================================")

def start_web_server(host='0.0.0.0', port=5000, detection_log_path=None,
                     detections_list=None, logs_list=None, detections_lock=None,
                     logs_lock=None, config=None):
    """Starts the Flask web server."""
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
    app.run(host=host, port=port, threaded=True)

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
def aggregation_thread_function(cooldown=30, bold_threshold=10):
    """
    Monitors detections_list and aggregates detections into summaries based on the cooldown period.
    
    Args:
        cooldown (int): Cooldown period in seconds after the last detection.
        bold_threshold (int): Threshold for bolding the number of detections.
    """
    last_detection_time = None
    current_aggregation = None  # To track ongoing aggregation
    
    while True:
        time.sleep(1)  # Check every second

        with app.config['detections_lock']:
            if app.config['detections_list']:
                # There are new detections
                for detection in app.config['detections_list']:
                    timestamp = datetime.strptime(detection['timestamp'], '%Y-%m-%d %H:%M:%S')
                    confidence = detection['confidence']
                    
                    if current_aggregation is None:
                        # Start a new aggregation
                        current_aggregation = {
                            'count': 1,
                            'first_timestamp': timestamp,
                            'latest_timestamp': timestamp,
                            'lowest_confidence': confidence,
                            'highest_confidence': confidence
                        }
                        # Create the summary string
                        summary = f"👀 Human detected {current_aggregation['count']} times within {cooldown} seconds! " \
                                  f"First seen: {current_aggregation['first_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}, " \
                                  f"Latest: {current_aggregation['latest_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}, " \
                                  f"Lowest confidence: {current_aggregation['lowest_confidence']:.2f}, " \
                                  f"Highest confidence: {current_aggregation['highest_confidence']:.2f}"
                        with aggregated_lock:
                            aggregated_detections_list.appendleft({'summary': summary})
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
                            else:
                                # This should not happen, but in case
                                aggregated_detections_list.appendleft({'summary': summary})
                
                # Update the last_detection_time to now
                last_detection_time = time.time()
    
                # Clear the detections_list after processing
                app.config['detections_list'].clear()
    
        # Check if cooldown period has passed since the last detection
        if current_aggregation and last_detection_time:
            if (time.time() - last_detection_time) >= cooldown:
                # Finalize the current aggregation by keeping it as is
                # Reset the aggregation to allow new aggregations
                current_aggregation = None
                last_detection_time = None

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
            <li>{{ detection.summary | safe }}</li>
        {% endfor %}
    </ul>

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

# Start the aggregation thread
aggregation_thread = threading.Thread(target=aggregation_thread_function, daemon=True)
aggregation_thread.start()

if __name__ == '__main__':
    # Example parameters; adjust as needed
    start_web_server(
        host='0.0.0.0',
        port=5000,
        detection_log_path='path/to/correct_detection_log_file.log',  # Replace with actual path
        detections_list=[],  # Initialize empty list or pass existing detections
        logs_list=[],        # Initialize empty list or pass existing logs
        detections_lock=threading.Lock(),
        logs_lock=threading.Lock()
    )