# web_server.py
# (Updated Oct 11, 2024)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Web server module for real-time YOLOv8 detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from datetime import datetime
import threading
import time
from flask import Flask, Response, render_template_string, stream_with_context, request, jsonify
import cv2
import logging
from web_graph import generate_detection_graph

# Configure logging for the web server
logger = logging.getLogger('web_server')
logger.setLevel(logging.INFO)

app = Flask(__name__)

# Global variables to hold the output frame and a lock for thread safety
output_frame = None
frame_lock = threading.Lock()

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

# def generate_frames():
#     """Generator function that yields frames in byte format for streaming."""
#     global output_frame
#     max_fps = app.config.get('webserver_max_fps', 10)  # Default to 10 FPS if not set
#     frame_interval = 1.0 / max_fps
#     last_frame_time = time.time()
#     while True:
#         with frame_lock:
#             if output_frame is None:
#                 # Sleep briefly to prevent 100% CPU utilization
#                 time.sleep(0.01)
#                 continue
#             # Limit the frame rate
#             current_time = time.time()
#             elapsed_time = current_time - last_frame_time
#             if elapsed_time < frame_interval:
#                 time.sleep(frame_interval - elapsed_time)
#             last_frame_time = current_time
            
#             # Encode the frame in JPEG format
#             ret, jpeg = cv2.imencode('.jpg', output_frame)

#             # Encode the frame in JPEG format with lower quality (adjust quality as needed)
#             # ret, jpeg = cv2.imencode('.jpg', output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

#             if not ret:
#                 continue
#             frame_bytes = jpeg.tobytes()
#         # Yield the output frame in byte format
#         # yield (b'--frame\r\n'
#         #        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         try:
#             yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         except GeneratorExit:
#             # Client disconnected
#             break
#         except Exception as e:
#             logger.error(f"Error in streaming frames: {e}")
#             break


# def generate_frames():
#     """Generator function that yields frames in byte format for streaming."""
#     global output_frame
#     while True:
#         with frame_lock:
#             if output_frame is None:
#                 # Sleep briefly to prevent 100% CPU utilization
#                 time.sleep(0.01)
#                 continue
#             # Encode the frame in JPEG format
#             ret, jpeg = cv2.imencode('.jpg', output_frame)
#             if not ret:
#                 continue
#             frame_bytes = jpeg.tobytes()
#         # Yield the output frame in byte format
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         # Add a small sleep to control frame rate and reduce CPU usage
#         time.sleep(0.05)

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

@app.route('/api/detections')
def get_detections():
    with app.config['detections_lock']:
        detections = list(app.config['detections_list'])

    detections_data = []
    for det in detections:
        detection_info = {
            'frame_count': int(det['frame_count']),
            'timestamp': str(det['timestamp']),
            'coordinates': [int(coord) for coord in det['coordinates']],
            'confidence': float(det['confidence'])
        }
        detections_data.append(detection_info)

    return jsonify(detections_data)
    # Alternatively, return detections directly if they are already in the desired format
    # return jsonify(detections)

@app.route('/api/logs')
def get_logs():
    with app.config['logs_lock']:
        logs = list(app.config['logs_list'])
    return jsonify(logs)

@app.route('/')
def index():
    """Homepage to display video streaming."""
    with app.config['detections_lock']:
        detections = list(app.config['detections_list'])
    with app.config['logs_lock']:
        logs = list(app.config['logs_list'])

    # Get the selected time range from query parameters
    hours = request.args.get('hours', default=None, type=int)
    graph_image = None
    if hours:
        detection_log_path = app.config['detection_log_path']
        graph_image = generate_detection_graph(hours, detection_log_path)

    # AJAX-based polling;
    # no more this:
    # <meta http-equiv="refresh" content="5">

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
            <li>
                <strong>Frame:</strong> {{ detection.frame_count }},
                <strong>Timestamp:</strong> {{ detection.timestamp }},
                <strong>Coordinates:</strong> {{ detection.coordinates }},
                <strong>Confidence:</strong> {{ detection.confidence|round(2) }}
            </li>
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
                        li.innerHTML = `<strong>Frame:</strong> ${detection.frame_count},
                                        <strong>Timestamp:</strong> ${detection.timestamp},
                                        <strong>Coordinates:</strong> ${detection.coordinates},
                                        <strong>Confidence:</strong> ${detection.confidence.toFixed(2)}`;
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
