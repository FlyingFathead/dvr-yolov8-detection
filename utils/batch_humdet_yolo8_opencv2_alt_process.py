# batch_humdet_yolo8_opencv2_alt_process.py

import torch
import cv2
import csv
import os
import numpy as np
import logging
import sys
import argparse
from ultralytics import YOLO
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SKIP_RESCALE = True
PROGRESS_INTERVAL = 1
DEFAULT_CONF_THRESHOLD = 0.8
DEFAULT_FRAME_FORMAT = 'jpg'
DRAW_RECTANGLES = True
CREATE_VIDEO = True
DEFAULT_MODEL_VARIANT = 'yolov8m'  # Use Medium model variant for better balance

torch.backends.cudnn.benchmark = True
cudnn_version = torch.backends.cudnn.version()
logging.info(f'Using cuDNN version: {cudnn_version}')

def load_model(model_variant=DEFAULT_MODEL_VARIANT):
    try:
        model = YOLO(model_variant)
        model.to('cuda')
        model.half()  # Use FP16 precision
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

model = load_model(DEFAULT_MODEL_VARIANT)

def save_detection_frame(frame, detections, frame_count, frame_output_dir, video_filename, frame_format, draw_rectangles):
    for detection in detections:
        x1, y1, x2, y2, confidence, class_idx = detection
        if draw_rectangles:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'Person: {confidence:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            logging.info(f'Drew rectangle at: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)}) with label: {label}')
    
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
    global model

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_update_interval = max(1, int(total_frames * (PROGRESS_INTERVAL / 100)))

    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    logging.info(f'Starting processing: {video_path}')
    logging.info(f'Frame format: {frame_format}')

    total_preprocess_time = 0
    total_inference_time = 0
    total_postprocess_time = 0

    with open(output_log, mode='w', newline='') as csv_file:
        fieldnames = ['timestamp', 'frame', 'x1', 'y1', 'x2', 'y2', 'confidence']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                start_preprocess = time.time()

                # Downscale the frame for faster processing
                frame = cv2.resize(frame, (1280, 720))

                # Define Regions of Interest (ROIs) if known
                # Example: rois = [(x, y, w, h), (x, y, w, h), ...]
                rois = [(0, 0, 1280, 720)]  # Adjust based on your specific areas of interest

                end_preprocess = time.time()
                start_inference = time.time()

                detections = []
                for roi in rois:
                    x, y, w, h = roi
                    roi_frame = frame[y:y+h, x:x+w]
                    results = model.predict(source=roi_frame, imgsz=[1280, 720], conf=conf_threshold, classes=[0], verbose=False)
                    roi_detections = results[0].boxes.data.cpu().numpy()
                    for det in roi_detections:
                        detections.append([det[0]+x, det[1]+y, det[2]+x, det[3]+y, det[4], det[5]])

                end_inference = time.time()
                start_postprocess = time.time()

                if detections:
                    timestamp = frame_count / frame_rate
                    for detection in detections:
                        writer.writerow({
                            'timestamp': timestamp,
                            'frame': frame_count,
                            'x1': detection[0],
                            'y1': detection[1],
                            'x2': detection[2],
                            'y2': detection[3],
                            'confidence': detection[4]
                        })
                    if frame_output_dir:
                        output_path = save_detection_frame(frame, detections, frame_count, frame_output_dir, video_filename, frame_format, draw_rectangles)
                        logging.info(f'Detection in {video_path}: frame {frame_count}, timestamp {timestamp:.2f}, confidence {detection[4]}, saved to {output_path}')
                
                end_postprocess = time.time()

                frame_count += 1
                total_preprocess_time += (end_preprocess - start_preprocess)
                total_inference_time += (end_inference - start_inference)
                total_postprocess_time += (end_postprocess - start_postprocess)

                if frame_count % progress_update_interval == 0:
                    percentage_complete = (frame_count / total_frames) * 100
                    logging.info(f'Processing {video_path}: {percentage_complete:.2f}% complete')

        except KeyboardInterrupt:
            logging.info("Process interrupted by user")
            cap.release()
            sys.exit(0)

    cap.release()
    logging.info(f'Finished processing: {video_path}')
    logging.info(f'Average preprocessing time per frame: {total_preprocess_time / frame_count:.2f} seconds')
    logging.info(f'Average inference time per frame: {total_inference_time / frame_count:.2f} seconds')
    logging.info(f'Average postprocessing time per frame: {total_postprocess_time / frame_count:.2f} seconds')

    if create_video:
        create_video_from_frames(frame_output_dir, video_filename, frame_format)

def process_video_file(video_file, video_dir, output_dir, target_resolution, conf_threshold, frame_output_dir, frame_format, draw_rectangles, create_video):
    video_path = os.path.join(video_dir, video_file)
    log_file = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_detections.csv")
    detect_humans(video_path, log_file, target_resolution, conf_threshold, frame_output_dir, frame_format, draw_rectangles, create_video)

def log_configuration(video_dir, conf_threshold, skip_rescale, model_variant, frame_format, draw_rectangles, create_video, total_videos):
    logging.info("Starting batch human detection processing with the following configuration:")
    logging.info(f"Video directory: {video_dir}")
    logging.info(f"Confidence threshold: {conf_threshold}")
    logging.info(f"Model variant: {model_variant}")
    logging.info(f"Saved frame file format: {frame_format}")
    logging.info(f"Draw rectangles: {draw_rectangles}")
    logging.info(f"Create video: {create_video}")
    logging.info(f"Total video files to process: {total_videos}")

def main():
    parser = argparse.ArgumentParser(description='Batch human detection using YOLOv8 and OpenCV.')
    parser.add_argument('video_dir', help='Directory containing video files for processing')
    parser.add_argument('--conf_threshold', type=float, default=DEFAULT_CONF_THRESHOLD, help='Confidence threshold for detection')
    parser.add_argument('--model_variant', type=str, default=DEFAULT_MODEL_VARIANT, help='YOLOv8 model variant to use (e.g., yolov8n, yolov8m, yolov8l)')
    parser.add_argument('--frame_format', type=str, default=DEFAULT_FRAME_FORMAT, choices=['png', 'jpg'], help='Format for saving detection frames (default: jpg)')
    parser.add_argument('--draw_rectangles', action='store_true', default=DRAW_RECTANGLES, help='Draw rectangles around detections')
    parser.add_argument('--create_video', action='store_true', default=CREATE_VIDEO, help='Create video from detection frames')
    args = parser.parse_args()

    video_dir = args.video_dir
    conf_threshold = args.conf_threshold
    model_variant = args.model_variant
    frame_format = args.frame_format
    draw_rectangles = args.draw_rectangles
    create_video = args.create_video

    global model
    model = load_model(model_variant)

    output_dir = os.path.join(video_dir, "detection_logs")
    frame_output_dir = os.path.join(video_dir, "detection_frames")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frame_output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    total_videos = len(video_files)

    log_configuration(video_dir, conf_threshold, SKIP_RESCALE, model_variant, frame_format, draw_rectangles, create_video, total_videos)

    try:
        for idx, video_file in enumerate(video_files):
            logging.info(f'Processing video {idx + 1} of {total_videos}: {video_file}')
            process_video_file(video_file, video_dir, output_dir, None, conf_threshold, frame_output_dir, frame_format, draw_rectangles, create_video)
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
