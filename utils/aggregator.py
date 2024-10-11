# utils/aggregator.py

import threading
import time
import json
import os
from datetime import datetime
from collections import deque

class DetectionAggregator:
    def __init__(self, log_directory, interval=60):
        """
        Initializes the DetectionAggregator.

        Args:
            log_directory (str): Directory where aggregated logs will be stored.
            interval (int): Aggregation interval in seconds (default is 60 seconds).
        """
        self.log_directory = log_directory
        self.interval = interval  # in seconds
        self.detections = deque()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        # Ensure the log directory exists
        os.makedirs(self.log_directory, exist_ok=True)
        self.log_file_path = os.path.join(self.log_directory, 'aggregated_detections.log')

        # Start the aggregation thread
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def add_detection(self, confidence):
        """
        Adds a detection confidence score to the aggregator.

        Args:
            confidence (float): Confidence score of the detection.
        """
        with self.lock:
            self.detections.append(confidence)

    def run(self):
        """
        Periodically aggregates detections and logs the summary statistics.
        """
        while not self.stop_event.is_set():
            time.sleep(self.interval)
            self.aggregate_and_log()

    def aggregate_and_log(self):
        """
        Aggregates the detections and logs the summary statistics.
        """
        with self.lock:
            if not self.detections:
                # No detections in this interval
                return

            count = len(self.detections)
            highest_conf = max(self.detections)
            lowest_conf = min(self.detections)
            self.detections.clear()

        # Create a timestamp for the aggregation
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Prepare the log entry
        log_entry = {
            'timestamp': timestamp,
            'total_detections': count,
            'highest_confidence': highest_conf,
            'lowest_confidence': lowest_conf
        }

        # Write the log entry as a JSON line
        try:
            with open(self.log_file_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            # Handle exceptions (e.g., log to a separate error log)
            print(f"Failed to write aggregated detection log: {e}")

    def stop(self):
        """
        Stops the aggregation thread.
        """
        self.stop_event.set()
        self.thread.join()
