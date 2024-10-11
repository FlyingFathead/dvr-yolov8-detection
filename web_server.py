# web_server.py
# (Updated Oct 11, 2024)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Web server module for real-time YOLOv8 detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import threading
import time
from flask import Flask, Response, render_template_string, stream_with_context, request
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

    return render_template_string('''
        <html>
        <head>
            <title>Real-time Human Detection</title>
            <meta http-equiv="refresh" content="5">
        </head>
        <body>
            <h1>Real-time Human Detection</h1>
            <img src="{{ url_for('video_feed') }}" width="100%">

            <h2>Latest Detections</h2>
            <ul>
            {% for detection in detections %}
                <li>
                    Frame {{ detection.frame_count }}, 
                    Timestamp {{ detection.timestamp }}, 
                    Coordinates: {{ detection.coordinates }}, 
                    Confidence: {{ detection.confidence|round(2) }}
                </li>
            {% endfor %}
            </ul>

            <h2>Backend Logs</h2>
            <ul>
            {% for log in logs %}
                <li>{{ log }}</li>
            {% endfor %}
            </ul>
                                  
            <h2>Detection Graphs</h2>
            <form action="/" method="get">
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
                <h3>Detections Over Time</h3>
                <img src="data:image/png;base64,{{ graph_image }}">
            {% endif %}

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
