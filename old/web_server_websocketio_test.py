# web_server.py
# (Updated Oct 11, 2024)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Web server module for real-time YOLOv8 detection using Flask-SocketIO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import eventlet  # Import eventlet for asynchronous operations
eventlet.monkey_patch()  # Monkey patch to make standard library cooperative

import threading
import time
from flask import Flask, Response, render_template_string, stream_with_context, request
from flask_socketio import SocketIO, emit
import cv2
import logging
from web_graph import generate_detection_graph

# Configure logging for the web server
logger = logging.getLogger('web_server')
logger.setLevel(logging.INFO)

# Initialize Flask app and SocketIO with eventlet
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')  # Specify async_mode

# Global variables to hold the output frame and a lock for thread safety
output_frame = None
frame_lock = threading.Lock()

def start_web_server(host='0.0.0.0', port=5000, detection_log_path=None, detections_list=None, logs_list=None, detections_lock=None, logs_lock=None):
    """Starts the Flask web server with SocketIO."""
    logger.info(f"Starting web server at http://{host}:{port}")
    app.config['detection_log_path'] = detection_log_path
    app.config['detections_list'] = detections_list if detections_list is not None else []
    app.config['logs_list'] = logs_list if logs_list is not None else []
    app.config['detections_lock'] = detections_lock if detections_lock is not None else threading.Lock()
    app.config['logs_lock'] = logs_lock if logs_lock is not None else threading.Lock()
    
    # Suppress Flask's default logging if necessary
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Run the app with SocketIO using eventlet
    socketio.run(app, host=host, port=port)

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

@app.route('/')
def index():
    """Homepage to display video streaming and real-time updates."""
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

    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Human Detection</title>
    <style>
        /* Basic Styling */
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
            <option value="8760">Last Year</option>
        </select>
        <input type="submit" value="View Graph">
    </form>

    <!-- Include the graph image or a message if no data is available -->
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

    <!-- Include Socket.IO -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.1/socket.io.min.js" integrity="sha512-LJx1h+ebxfJ1ZHiEu/JnpAs9dHR39bIlUB+IxQwvqjzHLKXhW1o/IT+CqJqMQJQBSraFs5UYTZ5UpWMFTp4sEA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script>
        // Initialize SocketIO
        const socket = io();

        // Listen for new detection events
        socket.on('new_detection', function(detection) {
            const detectionsList = document.getElementById('detections-list');
            const li = document.createElement('li');
            li.innerHTML = `<strong>Frame:</strong> ${detection.frame_count},
                            <strong>Timestamp:</strong> ${detection.timestamp},
                            <strong>Coordinates:</strong> ${detection.coordinates},
                            <strong>Confidence:</strong> ${detection.confidence.toFixed(2)}`;
            detectionsList.prepend(li); // Add to the top of the list

            // Optional: Limit the number of displayed detections
            if (detectionsList.children.length > 50) {
                detectionsList.removeChild(detectionsList.lastChild);
            }
        });

        // Listen for new log events
        socket.on('new_log', function(log) {
            const logsList = document.getElementById('logs-list');
            const li = document.createElement('li');
            li.textContent = log;
            logsList.prepend(li); // Add to the top of the list

            // Optional: Limit the number of displayed logs
            if (logsList.children.length > 100) {
                logsList.removeChild(logsList.lastChild);
            }
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
                    let graphContainer = document.getElementById('detection-graph');
                    if (!graphContainer) {
                        graphContainer = document.createElement('div');
                        graphContainer.id = 'detection-graph';
                        document.body.appendChild(graphContainer);
                    }
                    graphContainer.innerHTML = '<h3>Detections Over Time</h3>' + html;
                })
                .catch(error => {
                    console.error('Error fetching graph:', error);
                    // Display the error message in the graph container
                    let graphContainer = document.getElementById('detection-graph');
                    if (!graphContainer) {
                        graphContainer = document.createElement('div');
                        graphContainer.id = 'detection-graph';
                        document.body.appendChild(graphContainer);
                    }
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
        logging.info("User tried to request a detection data graph, but it was not available.")
        # Return an HTML snippet with a styled message
        return '<p style="color: red; font-weight: bold;">No detection data available for this time range.</p>'
    return f'<img src="data:image/png;base64,{image_base64}">'

def new_detection(detection):
    """Emit a new_detection event to all connected clients."""
    socketio.emit('new_detection', {
        'frame_count': detection['frame_count'],
        'timestamp': detection['timestamp'],
        'coordinates': detection['coordinates'],
        'confidence': detection['confidence']
    })

def new_log(log_message):
    """Emit a new_log event to all connected clients."""
    socketio.emit('new_log', log_message)

def add_new_detection(detection):
    """Add a new detection to the list and emit an event."""
    with app.config['detections_lock']:
        app.config['detections_list'].append(detection)
    new_detection(detection)  # Emit the event

def add_new_log(log_message):
    """Add a new log to the list and emit an event."""
    with app.config['logs_lock']:
        app.config['logs_list'].append(log_message)
    new_log(log_message)  # Emit the event

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
        add_new_detection(detection)
        add_new_log(f'Detection {frame_count} added.')

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
    
    # Start the simulation as a background task using SocketIO's method
    # Note: Move this before the server starts or integrate it properly
    # To ensure it starts correctly, place it within a SocketIO event
    # Here's a safe way to start it after the server is running
    @socketio.on('connect')
    def handle_connect():
        logger.info("Client connected. Starting simulation.")
        socketio.start_background_task(simulate_detection_and_logging)
