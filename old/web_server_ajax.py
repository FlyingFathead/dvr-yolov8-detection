# web_server.py
# (Updated Oct 11, 2024)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Web server module for real-time YOLOv8 detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import threading
import time
from flask import Flask, Response, render_template_string, stream_with_context, request
# for AJAX-based polling
from flask import jsonify
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

def start_web_server(host='0.0.0.0', port=5000, detection_log_path=None, detections_list=None, logs_list=None, detections_lock=None, logs_lock=None):
    """Starts the Flask web server."""
    logger.info(f"Starting web server at http://{host}:{port}")
    app.config['detection_log_path'] = detection_log_path
    app.config['detections_list'] = detections_list
    app.config['logs_list'] = logs_list
    app.config['detections_lock'] = detections_lock
    app.config['logs_lock'] = logs_lock
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
    while True:
        with frame_lock:
            if output_frame is None:
                # Sleep briefly to prevent 100% CPU utilization
                time.sleep(0.01)
                continue
            # Encode the frame in JPEG format
            ret, jpeg = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()
        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Add a small sleep to control frame rate and reduce CPU usage
        time.sleep(0.05)

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
    detections_data = [{
        'frame_count': det.frame_count,
        'timestamp': det.timestamp,
        'coordinates': det.coordinates,
        'confidence': det.confidence
    } for det in detections]
    return jsonify(detections_data)

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

    # # Get the selected time range from query parameters
    # hours = request.args.get('hours', default=None, type=int)
    # graph_image = None
    # if hours:
    #     graph_image = generate_detection_graph(hours)

    # # Get the selected time range from query parameters
    # hours = request.args.get('hours', default=None, type=int)
    # graph_image = None
    # if hours:
    #     graph_image = generate_detection_graph(hours, app.config['detection_timestamps'], app.config['timestamps_lock'])

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
        /* Optional: Add some basic styling */
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1, h2 {
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
    </style>
</head>
<body>
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

        // Function to fetch and update the detection graph
        function fetchGraph(hours) {
            // Optionally implement if dynamic graph updates are needed
            // For simplicity, this example reloads the page upon form submission
        }

        // Set intervals to fetch data every 5 seconds
        setInterval(() => {
            fetchDetections();
            fetchLogs();
        }, 5000); // 5000 milliseconds = 5 seconds

        // Initial fetch when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            fetchDetections();
            fetchLogs();
        });

        // Optional: Handle form submission via AJAX to avoid page reload
        document.getElementById('graph-form').addEventListener('submit', function(event) {
            event.preventDefault(); // Prevent the default form submission
            const hours = document.getElementById('hours').value;
            // Fetch the graph image via AJAX or simply reload the graph section
            fetch(`/detection_graph/${hours}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.text();
                })
                .then(html => {
                    const graphContainer = document.getElementById('detection-graph') || document.createElement('div');
                    graphContainer.id = 'detection-graph';
                    graphContainer.innerHTML = '<h3>Detections Over Time</h3>' + html;
                    document.body.appendChild(graphContainer);
                })
                .catch(error => {
                    console.error('Error fetching graph:', error);
                    alert('Failed to load detection graph.');
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
        return 'No detection data available for this time range.', 404
    return f'<img src="data:image/png;base64,{image_base64}">'
