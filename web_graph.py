# web_graph.py

import logging
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import time
import os
import matplotlib.dates as mdates

def generate_detection_graph(time_range_hours, detection_log_path):
    """Generates a histogram of detections over the specified time range."""
    now = datetime.now()
    start_time = now - timedelta(hours=time_range_hours)

    timestamps = []

    # Read the detection log file
    if not os.path.exists(detection_log_path):
        logging.warning(f"Detection log file not found: {detection_log_path}")
        return None

    with open(detection_log_path, 'r') as f:
        for line in f:
            try:
                if 'Timestamp ' in line:
                    timestamp_str = line.split('Timestamp ')[1].split(',')[0].strip()
                    detection_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    if detection_time >= start_time:
                        timestamps.append(detection_time)
            except Exception as e:
                logging.exception(f"Error parsing line: {line}")
                continue

    if not timestamps:
        logging.info("No detection data available for the selected time range.")
        return None  # No data to plot

    # Create the histogram
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Set binning
    if time_range_hours <= 24:
        # For time ranges up to 24 hours, use hourly bins
        bin_width = timedelta(hours=1)
        locator = mdates.HourLocator()
        formatter = mdates.DateFormatter('%H:%M')
    elif time_range_hours <= 168:  # Up to one week
        # Use 6-hour bins
        bin_width = timedelta(hours=6)
        locator = mdates.HourLocator(interval=6)
        formatter = mdates.DateFormatter('%b %d %H:%M')
    else:
        # For longer time ranges, use daily bins
        bin_width = timedelta(days=1)
        locator = mdates.DayLocator()
        formatter = mdates.DateFormatter('%b %d')

    # Compute the number of bins
    num_bins = int((now - start_time) / bin_width)
    bins = [start_time + i * bin_width for i in range(num_bins + 1)]

    # Plot histogram
    plt.hist(timestamps, bins=bins, edgecolor='black')

    # Set the x-axis formatter and locator
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    plt.title(f'Detections in the Last {time_range_hours} Hours')
    plt.xlabel('Time')
    plt.ylabel('Number of Detections')
    plt.tight_layout()

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('ascii')
    return image_base64

# # web_graph.py

# import logging
# import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend
# import matplotlib.pyplot as plt
# import io
# import base64
# from datetime import datetime, timedelta
# import time
# import os

# def generate_detection_graph(time_range_hours, detection_log_path):
#     """Generates a histogram of detections over the specified time range."""
#     now = datetime.now()
#     start_time = now - timedelta(hours=time_range_hours)

#     timestamps = []

#     # Read the detection log file
#     if not os.path.exists(detection_log_path):
#         # No log file exists
#         return None

#     with open(detection_log_path, 'r') as f:
#         for line in f:
#             # Example line:
#             # 2023-10-11 14:55:12,INFO - Detection: Frame 123, Timestamp 2023-10-11 14:55:12, Coordinates: (x1, y1), (x2, y2), Confidence: 0.90
#             try:
#                 # Extract the timestamp after 'Timestamp '
#                 if 'Timestamp ' in line:
#                     timestamp_str = line.split('Timestamp ')[1].split(',')[0].strip()
#                     detection_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
#                     if detection_time >= start_time:
#                         timestamps.append(detection_time)
#             except Exception as e:
#                 logging.exception(f"Error parsing line: {line}")                
#                 # Skip lines that cannot be parsed
#                 continue

#     if not timestamps:
#         return None  # No data to plot

#     # Create the histogram
#     plt.figure(figsize=(10, 4))
#     plt.hist(timestamps, bins=24, edgecolor='black')
#     plt.title(f'Detections in the Last {time_range_hours} Hours')
#     plt.xlabel('Time')
#     plt.ylabel('Number of Detections')
#     plt.tight_layout()

#     # Save the plot to a bytes buffer
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
#     buf.seek(0)
#     image_base64 = base64.b64encode(buf.read()).decode('ascii')
#     return image_base64
