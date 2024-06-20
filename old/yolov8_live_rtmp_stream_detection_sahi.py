import cv2
import torch
import logging
import time
import pyttsx3
import os
import numpy as np
from ultralytics import YOLO
from threading import Thread, Event, Lock

# Version number
version_number = 0.125

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default configuration
DEFAULT_CONF_THRESHOLD = 0.7
DEFAULT_MODEL_VARIANT = 'yolov8m'
DRAW_RECTANGLES = True
SAVE_DETECTIONS = True
IMAGE_FORMAT = 'jpg'
SAVE_DIR = '/mnt/yolo_detections'
RETRY_DELAY = 2
MAX_RETRIES = 10
RESCALE_INPUT = False
TARGET_HEIGHT = 1080

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_lock = Lock()

# Load the YOLOv8 model
def load_model(model_variant=DEFAULT_MODEL_VARIANT):
    try:
        model = YOLO(model_variant)
        model.to('cuda')
        return model
    except Exception as e:
        logging.error(f"Error loading model {model_variant}: {e}")
        raise

# Initialize model
model = load_model(DEFAULT_MODEL_VARIANT)

# Function to resize while maintaining aspect ratio
def resize_frame(frame, target_height):
    """Resize the frame to the target height while maintaining the aspect ratio."""
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    resized_frame = cv2.resize(frame, (new_width, target_height))
    return resized_frame

# Function to announce detection
def announce_detection(stop_event, tts_lock):
    """Announce human detection using text-to-speech."""
    while not stop_event.is_set():
        with tts_lock:
            logging.info("Human detected!")
            tts_engine.say("Human detected!")
            tts_engine.runAndWait()
        time.sleep(1)

# Function to save detection image
def save_detection_image(frame, detection_count):
    try:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(SAVE_DIR, f"detection_{detection_count}_{timestamp}.{IMAGE_FORMAT}")
        cv2.imwrite(filename, frame)
        logging.info(f"Saved detection image: {filename}")
    except Exception as e:
        logging.error(f"Error saving detection image: {e}")

# Function to run YOLOv8 on a frame and return predictions
def run_yolo_on_frame(frame, model, conf_threshold):
    results = model.predict(source=frame, conf=conf_threshold, classes=[0], verbose=False)
    return results[0].boxes.data.cpu().numpy()

# Function to manually slice the image
def slice_image(image, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio):
    height, width = image.shape[:2]
    y_steps = int(np.ceil(height / (slice_height * (1 - overlap_height_ratio))))
    x_steps = int(np.ceil(width / (slice_width * (1 - overlap_width_ratio))))

    slices = []
    for y in range(y_steps):
        for x in range(x_steps):
            start_y = int(y * slice_height * (1 - overlap_height_ratio))
            end_y = min(start_y + slice_height, height)
            start_x = int(x * slice_width * (1 - overlap_width_ratio))
            end_x = min(start_x + slice_width, width)

            slice_img = image[start_y:end_y, start_x:end_x]
            slices.append({
                "image": slice_img,
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y
            })
    return slices

# Compute IOU
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

# Merge two boxes
def merge_boxes(box1, box2):
    x1, y1, x2, y2, conf1, class1 = box1
    x1_p, y1_p, x2_p, y2_p, conf2, class2 = box2
    x1 = min(x1, x1_p)
    y1 = min(y1, y1_p)
    x2 = max(x2, x2_p)
    y2 = max(y2, y2_p)
    conf = max(conf1, conf2)
    return (x1, y1, x2, y2, conf, class1)

# Manual merge function
def merge_detections(detections, iou_threshold=0.5):
    merged_detections = []
    for i, det in enumerate(detections):
        if det is None:
            continue
        for j in range(i + 1, len(detections)):
            if detections[j] is None:
                continue
            iou = compute_iou(det[:4], detections[j][:4])
            if iou > iou_threshold:
                merged_det = merge_boxes(det, detections[j])
                detections[j] = None
                det = merged_det
        merged_detections.append(det)
    return merged_detections

# Process the incoming stream
def process_stream(stream_url, conf_threshold=DEFAULT_CONF_THRESHOLD, draw_rectangles=DRAW_RECTANGLES):
    """Process the video stream and detect humans."""
    cap = None
    retries = 0
    detection_count = 0

    while retries < MAX_RETRIES:
        cap = cv2.VideoCapture(f"ffmpeg -i {stream_url} -f rawvideo -pix_fmt bgr24 pipe:1", cv2.CAP_FFMPEG)
        if cap.isOpened():
            break
        else:
            logging.error(f"Unable to open video stream: {stream_url}. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            retries += 1

    if cap is None or not cap.isOpened():
        logging.error(f"Failed to open video stream: {stream_url} after {MAX_RETRIES} retries.")
        return

    logging.info(f"Starting video stream: {stream_url}")
    cv2.namedWindow('Real-time Human Detection', cv2.WINDOW_NORMAL)

    stop_event = Event()
    announce_thread = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from stream. Retrying...")
                time.sleep(1)
                continue

            start_time = time.time()
            if RESCALE_INPUT:
                resized_frame = resize_frame(frame, TARGET_HEIGHT)
            else:
                resized_frame = frame

            # Run detection using manual slicing on the resized frame
            slices = slice_image(resized_frame, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
            all_detections = []

            for sliced_image_info in slices:
                sliced_image = sliced_image_info["image"]
                slice_detections = run_yolo_on_frame(sliced_image, model, conf_threshold)
                for detection in slice_detections:
                    x1, y1, x2, y2, confidence, class_idx = detection
                    x1 += sliced_image_info["start_x"]
                    y1 += sliced_image_info["start_y"]
                    x2 += sliced_image_info["start_x"]
                    y2 += sliced_image_info["start_y"]
                    all_detections.append((x1, y1, x2, y2, confidence, class_idx))

            # Merge detections from all slices
            merged_detections = merge_detections(all_detections, iou_threshold=0.5)

            if merged_detections:
                detection_count += 1
                if announce_thread is None or not announce_thread.is_alive():
                    stop_event.clear()
                    announce_thread = Thread(target=announce_detection, args=(stop_event, tts_lock))
                    announce_thread.start()
                for detection in merged_detections:
                    x1, y1, x2, y2, confidence, class_idx = detection
                    if draw_rectangles:
                        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(resized_frame.shape[1], int(x2)), min(resized_frame.shape[0], int(y2))
                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'Person: {confidence:.2f}'
                        cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if SAVE_DETECTIONS:
                    save_detection_image(resized_frame, detection_count)
            else:
                if announce_thread is not None and announce_thread.is_alive():
                    stop_event.set()
                    announce_thread.join()
                    announce_thread = None

            cv2.imshow('Real-time Human Detection', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            end_time = time.time()
            logging.info(f'Processed frame in {end_time - start_time:.2f} seconds')

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_event.set()
        if announce_thread is not None:
            announce_thread.join()

if __name__ == "__main__":
    stream_url = 'rtmp://127.0.0.1:1935/live/stream'
    process_stream(stream_url)
