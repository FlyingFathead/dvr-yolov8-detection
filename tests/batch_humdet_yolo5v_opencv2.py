# batch_humdet_yolo8_opencv2.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/dvr-yolo5-detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

version_number = 0.12

import torch
import cv2
import csv
import os
import numpy as np
import logging
import sys
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# User-configurable progress update interval (percentage)
PROGRESS_INTERVAL = 5

# Default confidence threshold
DEFAULT_CONF_THRESHOLD = 0.7

# Variable to control whether rescaling is skipped
SKIP_RESCALE = True

# Default frame format; use 'png' for higher quality (less compression, etc), 'jpg' for smaller file size
DEFAULT_FRAME_FORMAT = 'jpg'

# Variable to control whether to draw rectangles around detections
DRAW_RECTANGLES = False

# Variable to control whether to create a video from frames
CREATE_VIDEO = True

# Function to load the model
# use i.e. 'yolov5s' or 'yolov5l'  
def load_model(model_variant='yolov5l'):
    model = torch.hub.load('ultralytics/yolov5', model_variant, pretrained=True)
    model.to('cuda')  # GPU acceleration
    return model

# Load YOLOv5 model
model = load_model('yolov5l')  # Initial default load

def letterbox_image(image, target_resolution):
    # Compute scale factors
    height, width = image.shape[:2]
    target_width, target_height = target_resolution

    scale = min(target_width / width, target_height / height)
    new_width, new_height = int(width * scale), int(height * scale)

    # Resize while maintaining aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height))

    # Padding to fit the target resolution
    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    # Add padding to the resized image
    letterboxed_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return letterboxed_image

def save_detection_frame(frame, detections, frame_count, frame_output_dir, video_filename, frame_format, draw_rectangles):
    for detection in detections:
        x_center, y_center, width, height, confidence, class_idx = detection
        if draw_rectangles:
            x1 = int((x_center - width / 2) * frame.shape[1])
            y1 = int((y_center - height / 2) * frame.shape[0])
            x2 = int((x_center + width / 2) * frame.shape[1])
            y2 = int((y_center + height / 2) * frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{model.names[int(class_idx)]}: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            logging.info(f'Drew rectangle at: ({x1}, {y1}), ({x2}, {y2}) with label: {label}')
    
    output_path = os.path.join(frame_output_dir, f'{video_filename}_frame_{frame_count}.{frame_format}')
    cv2.imwrite(output_path, frame)
    logging.info(f'Saved frame with detections to: {output_path}')
    return output_path

def create_video_from_frames(frame_output_dir, video_filename, frame_format, fps=1):
    frame_files = sorted([f for f in os.listdir(frame_output_dir) if f.endswith(frame_format)])
    if not frame_files:
        logging.warning(f'No frames found in {frame_output_dir} to create video.')
        return

    frame_path = os.path.join(frame_output_dir, frame_files[0])
    frame = cv2.imread(frame_path)
    height, width, layers = frame.shape

    video_output_path = os.path.join('detection_clips', f'{video_filename}.mp4')
    os.makedirs('detection_clips', exist_ok=True)

    video = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frame_output_dir, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)

    video.release()
    logging.info(f'Created video from frames: {video_output_path}')

def detect_humans(video_path, output_log, target_resolution, conf_threshold, frame_output_dir, frame_format, draw_rectangles, create_video):
    global model  # Use the globally defined model
    global SKIP_RESCALE  # Use the globally defined SKIP_RESCALE

    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_update_interval = max(1, int(total_frames * (PROGRESS_INTERVAL / 100)))

    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    logging.info(f'Starting processing: {video_path}')
    logging.info(f'Frame format: {frame_format}')
    # logging.info(f'Skip rescale: {SKIP_RESCALE}')  # Add logging here to verify the value

    # Open CSV to log detections
    with open(output_log, mode='w', newline='') as csv_file:
        fieldnames = ['timestamp', 'frame', 'x_center', 'y_center', 'width', 'height', 'confidence']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        try:
            # Loop through frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if SKIP_RESCALE:
                    frame_resized = frame
                else:
                    # Apply letterboxing to maintain aspect ratio
                    frame_resized = letterbox_image(frame, target_resolution)

                # Run detection on the resized frame
                results = model([frame_resized], size=640)
                detected = results.xywh[0]  # Extract bounding boxes

                # Check if any "person" is detected
                person_class_idx = 0  # YOLO class index for "person"
                detections = [d for d in detected if int(d[5].item()) == person_class_idx and d[4].item() >= conf_threshold]
                
                if detections:
                    timestamp = frame_count / frame_rate
                    for detection in detections:
                        writer.writerow({
                            'timestamp': timestamp,
                            'frame': frame_count,
                            'x_center': detection[0].item(),
                            'y_center': detection[1].item(),
                            'width': detection[2].item(),
                            'height': detection[3].item(),
                            'confidence': detection[4].item()
                        })
                    if frame_output_dir:
                        output_path = save_detection_frame(frame, detections, frame_count, frame_output_dir, video_filename, frame_format, draw_rectangles)
                        for detection in detections:
                            logging.info(f'Detection in {video_path}: frame {frame_count}, timestamp {timestamp:.2f}, confidence {detection[4].item():.2f}, saved to {output_path}')

                frame_count += 1

                # Log progress at the specified interval
                if frame_count % progress_update_interval == 0:
                    percentage_complete = (frame_count / total_frames) * 100
                    logging.info(f'Processing {video_path}: {percentage_complete:.2f}% complete')

        except KeyboardInterrupt:
            logging.info("Process interrupted by user")
            cap.release()
            sys.exit(0)

    cap.release()
    logging.info(f'Finished processing: {video_path}')

    # Create video from frames if enabled
    if create_video:
        create_video_from_frames(frame_output_dir, video_filename, frame_format)

def log_configuration(video_dir, conf_threshold, skip_rescale, model_variant, frame_format, draw_rectangles, create_video, total_videos):
    logging.info("Starting batch human detection processing with the following configuration:")
    logging.info(f"Video directory: {video_dir}")
    logging.info(f"Confidence threshold: {conf_threshold}")
    logging.info(f"Skip rescale: {skip_rescale}")
    logging.info(f"Model variant: {model_variant}")
    logging.info(f"Saved frame file format: {frame_format}")
    logging.info(f"Draw rectangles: {draw_rectangles}")
    logging.info(f"Create video: {create_video}")
    logging.info(f"Total video files to process: {total_videos}")

def main():
    global SKIP_RESCALE  # Declare SKIP_RESCALE as global at the beginning of the main function
    global DRAW_RECTANGLES  # Declare DRAW_RECTANGLES as global at the beginning of the main function
    global CREATE_VIDEO  # Declare CREATE_VIDEO as global at the beginning of the main function

    parser = argparse.ArgumentParser(description='Batch human detection using YOLOv5 and OpenCV.')
    parser.add_argument('video_dir', help='Directory containing video files for processing')
    parser.add_argument('--conf_threshold', type=float, default=DEFAULT_CONF_THRESHOLD, help='Confidence threshold for detection')
    parser.add_argument('--skip_rescale', action='store_true', default=SKIP_RESCALE, help='Skip rescaling of video frames')
    parser.add_argument('--model_variant', type=str, default='yolov5l', help='YOLOv5 model variant to use (e.g., yolov5s, yolov5m, yolov5l)')
    parser.add_argument('--frame_format', type=str, default=DEFAULT_FRAME_FORMAT, choices=['png', 'jpg'], help='Format for saving detection frames (default: jpg)')
    parser.add_argument('--draw_rectangles', action='store_true', default=DRAW_RECTANGLES, help='Draw rectangles around detections')
    parser.add_argument('--create_video', action='store_true', default=CREATE_VIDEO, help='Create video from detection frames')
    args = parser.parse_args()

    video_dir = args.video_dir
    conf_threshold = args.conf_threshold
    SKIP_RESCALE = args.skip_rescale
    model_variant = args.model_variant
    frame_format = args.frame_format
    DRAW_RECTANGLES = args.draw_rectangles
    CREATE_VIDEO = args.create_video

    global model
    model = load_model(model_variant)

    output_dir = os.path.join(video_dir, "detection_logs")
    frame_output_dir = os.path.join(video_dir, "detection_frames")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frame_output_dir, exist_ok=True)

    target_resolution = (1920, 1080)

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    total_videos = len(video_files)

    # Log configuration settings
    log_configuration(video_dir, conf_threshold, SKIP_RESCALE, model_variant, frame_format, DRAW_RECTANGLES, CREATE_VIDEO, total_videos)

    try:
        for idx, video_file in enumerate(video_files):
            logging.info(f'Processing video {idx + 1} of {total_videos}: {video_file}')
            process_video_file(video_file, video_dir, output_dir, target_resolution, conf_threshold, frame_output_dir, frame_format, DRAW_RECTANGLES, CREATE_VIDEO)
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(0)

def process_video_file(video_file, video_dir, output_dir, target_resolution, conf_threshold, frame_output_dir, frame_format, draw_rectangles, create_video):
    video_path = os.path.join(video_dir, video_file)
    log_file = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_detections.csv")
    detect_humans(video_path, log_file, target_resolution, conf_threshold, frame_output_dir, frame_format, draw_rectangles, create_video)

if __name__ == "__main__":
    main()
