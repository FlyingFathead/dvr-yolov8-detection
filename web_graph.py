# web_graph.py

import logging
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import os
import matplotlib.dates as mdates
import numpy as np

def generate_detection_graph(time_range_hours, detection_log_path):
    """Generates a histogram of detections over the specified time range with enhanced readability."""
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

    # Convert datetime to matplotlib's numerical format
    dates = mdates.date2num(timestamps)
    start_num = mdates.date2num(start_time)
    end_num = mdates.date2num(now)

    # Create the histogram
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Define bin width, label, locator, and formatter based on the time range
    if time_range_hours <= 1:
        # For up to 1 hour, use 1-minute bins
        bin_width = 1.0 / (24 * 60)  # One minute in days
        time_label = "Last Hour"
        bin_locator = mdates.MinuteLocator(interval=5)
        bin_formatter = mdates.DateFormatter('%H:%M')
    elif time_range_hours <= 3:
        # For up to 3 hours, use 1-minute bins
        bin_width = 1.0 / (24 * 60)  # One minute in days
        time_label = "Last 3 Hours"
        bin_locator = mdates.MinuteLocator(interval=15)
        bin_formatter = mdates.DateFormatter('%H:%M')
    elif time_range_hours <= 24:
        # For time ranges up to 24 hours, use 5-minute bins
        bin_width = 5.0 / (24 * 60)  # Five minutes in days
        time_label = "Last 24 Hours"
        bin_locator = mdates.HourLocator(interval=1)
        bin_formatter = mdates.DateFormatter('%H:%M')
    elif time_range_hours <= 168:  # Up to one week
        # Use 6-hour bins
        bin_width = 6.0 / 24  # Six hours in days
        time_label = "Last Week"
        bin_locator = mdates.DayLocator(interval=1)
        bin_formatter = mdates.DateFormatter('%b %d')
    elif time_range_hours <= 720:  # Up to one month
        # Use daily bins
        bin_width = 1  # One day
        time_label = "Last Month"
        bin_locator = mdates.DayLocator(interval=1)
        bin_formatter = mdates.DateFormatter('%b %d')
    elif time_range_hours <= 8760:  # Up to one year
        # Use weekly bins
        bin_width = 7  # Seven days
        time_label = "Last Year"
        bin_locator = mdates.MonthLocator(interval=1)
        bin_formatter = mdates.DateFormatter('%b')
    else:
        # For longer time ranges, use monthly bins
        bin_width = 30  # Approximately thirty days
        time_label = "Multiple Years"
        bin_locator = mdates.MonthLocator(interval=3)
        bin_formatter = mdates.DateFormatter('%b %Y')

    # Compute the number of bins and define bin edges
    if time_range_hours <= 24:
        # For up to 24 hours, use finer binning
        bins = np.arange(start_num, end_num + bin_width, bin_width)
    elif time_range_hours <= 168:
        # For up to one week, use 6-hour bins
        bins = np.arange(start_num, end_num + bin_width, bin_width)
    elif time_range_hours <= 720:
        # For up to one month, use daily bins
        bins = np.arange(start_num, end_num + bin_width, bin_width)
    elif time_range_hours <= 8760:
        # For up to one year, use weekly bins
        bins = np.arange(start_num, end_num + bin_width, bin_width)
    else:
        # For multiple years, use monthly bins
        bins = np.arange(start_num, end_num + bin_width, bin_width)

    # Plot histogram
    plt.hist(dates, bins=bins, edgecolor='black')

    # Set the x-axis formatter and locator
    ax.xaxis.set_major_locator(bin_locator)
    ax.xaxis.set_major_formatter(bin_formatter)

    # Set x-axis limits to ensure the graph spans exactly the desired time range
    ax.set_xlim(start_num, end_num)

    # Improve layout and readability
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Detections {time_label}')
    plt.xlabel('Time')
    plt.ylabel('Number of Detections')
    plt.tight_layout()

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)  # Increased DPI for better resolution
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
# import matplotlib.dates as mdates

# def generate_detection_graph(time_range_hours, detection_log_path):
#     """Generates a histogram of detections over the specified time range."""
#     now = datetime.now()
#     start_time = now - timedelta(hours=time_range_hours)

#     timestamps = []

#     # Read the detection log file
#     if not os.path.exists(detection_log_path):
#         logging.warning(f"Detection log file not found: {detection_log_path}")
#         return None

#     with open(detection_log_path, 'r') as f:
#         for line in f:
#             try:
#                 if 'Timestamp ' in line:
#                     timestamp_str = line.split('Timestamp ')[1].split(',')[0].strip()
#                     detection_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
#                     if detection_time >= start_time:
#                         timestamps.append(detection_time)
#             except Exception as e:
#                 logging.exception(f"Error parsing line: {line}")
#                 continue

#     if not timestamps:
#         logging.info("No detection data available for the selected time range.")
#         return None  # No data to plot

#     # Create the histogram
#     plt.figure(figsize=(12, 6))
#     ax = plt.gca()

#     # Set binning
#     if time_range_hours <= 24:
#         # For time ranges up to 24 hours, use hourly bins
#         bin_width = timedelta(hours=1)
#         locator = mdates.HourLocator()
#         formatter = mdates.DateFormatter('%H:%M')
#     elif time_range_hours <= 168:  # Up to one week
#         # Use 6-hour bins
#         bin_width = timedelta(hours=6)
#         locator = mdates.HourLocator(interval=6)
#         formatter = mdates.DateFormatter('%b %d %H:%M')
#     else:
#         # For longer time ranges, use daily bins
#         bin_width = timedelta(days=1)
#         locator = mdates.DayLocator()
#         formatter = mdates.DateFormatter('%b %d')

#     # Compute the number of bins
#     num_bins = int((now - start_time) / bin_width)
#     bins = [start_time + i * bin_width for i in range(num_bins + 1)]

#     # Plot histogram
#     plt.hist(timestamps, bins=bins, edgecolor='black')

#     # Set the x-axis formatter and locator
#     ax.xaxis.set_major_locator(locator)
#     ax.xaxis.set_major_formatter(formatter)

#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45, ha='right')

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

# # # web_graph.py

# # import logging
# # import matplotlib
# # matplotlib.use('Agg')  # Use a non-interactive backend
# # import matplotlib.pyplot as plt
# # import io
# # import base64
# # from datetime import datetime, timedelta
# # import time
# # import os

# # def generate_detection_graph(time_range_hours, detection_log_path):
# #     """Generates a histogram of detections over the specified time range."""
# #     now = datetime.now()
# #     start_time = now - timedelta(hours=time_range_hours)

# #     timestamps = []

# #     # Read the detection log file
# #     if not os.path.exists(detection_log_path):
# #         # No log file exists
# #         return None

# #     with open(detection_log_path, 'r') as f:
# #         for line in f:
# #             # Example line:
# #             # 2023-10-11 14:55:12,INFO - Detection: Frame 123, Timestamp 2023-10-11 14:55:12, Coordinates: (x1, y1), (x2, y2), Confidence: 0.90
# #             try:
# #                 # Extract the timestamp after 'Timestamp '
# #                 if 'Timestamp ' in line:
# #                     timestamp_str = line.split('Timestamp ')[1].split(',')[0].strip()
# #                     detection_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
# #                     if detection_time >= start_time:
# #                         timestamps.append(detection_time)
# #             except Exception as e:
# #                 logging.exception(f"Error parsing line: {line}")                
# #                 # Skip lines that cannot be parsed
# #                 continue

# #     if not timestamps:
# #         return None  # No data to plot

# #     # Create the histogram
# #     plt.figure(figsize=(10, 4))
# #     plt.hist(timestamps, bins=24, edgecolor='black')
# #     plt.title(f'Detections in the Last {time_range_hours} Hours')
# #     plt.xlabel('Time')
# #     plt.ylabel('Number of Detections')
# #     plt.tight_layout()

# #     # Save the plot to a bytes buffer
# #     buf = io.BytesIO()
# #     plt.savefig(buf, format='png')
# #     plt.close()
# #     buf.seek(0)
# #     image_base64 = base64.b64encode(buf.read()).decode('ascii')
# #     return image_base64
