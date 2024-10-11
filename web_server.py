# web_server.py
# (Updated Oct 11, 2024)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Web server module for real-time YOLOv8 detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import threading
import time
from flask import Flask, Response, render_template_string, stream_with_context
import cv2
import logging

# Configure logging for the web server
logger = logging.getLogger('web_server')
logger.setLevel(logging.INFO)

app = Flask(__name__)

# Global variables to hold the output frame and a lock for thread safety
output_frame = None
frame_lock = threading.Lock()

def start_web_server(host='0.0.0.0', port=5000):
    """Starts the Flask web server."""
    logger.info(f"Starting web server at http://{host}:{port}")
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
    return render_template_string('''
        <html>
        <head>
            <title>Real-time Human Detection</title>
        </head>
        <body>
            <h1>Real-time Human Detection</h1>
            <img src="{{ url_for('video_feed') }}" width="100%">
        </body>
        </html>
    ''')
