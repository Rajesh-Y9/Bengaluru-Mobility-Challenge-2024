import cv2
import numpy as np
from collections import defaultdict
import json
import os
from tqdm import tqdm
from boxmot import BoTSORT
from ultralytics import YOLO
from ultralytics import RTDETR
from pathlib import Path
import torch

if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available, using CPU")
    device = torch.device("cpu")

# Initialize the tracker
tracker = BoTSORT(
    model_weights=Path('clip_veri.pt'),  # ReID model to use
    device='cuda:0',
    fp16=False,
)

# Initialize YOLO model
yolo_model = YOLO('tracking_models/yolov10x.pt')

def process_videos(input_json):
    with open(input_json, 'r') as f:
        data = json.load(f)

    cam_id = list(data.keys())[0]
    video_paths = [
        data[cam_id]['Vid_1'], data[cam_id]['Vid_2']
    ]

    tracking_data = defaultdict(lambda: {'frameid': [], 'class': [], 'box': [], 'tracklets': []})
    frame_counter = 0

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        
        # Enable hardware acceleration
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        cap.set(cv2.CAP_PROP_HW_DEVICE, 0)
        
        if not cap.isOpened():
            print(f"Error: Could not open input video: {video_path}")
            continue

        # Get the original frame rate and total frames
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine if we need to skip frames
        skip_frames = original_fps > 30
        if skip_frames:
            frame_interval = 2
        else:
            frame_interval = 1

        print(f"Processing {video_path} - Original FPS: {original_fps:.2f}, {'Skipping frames' if skip_frames else 'Processing all frames'}")

        # Use tqdm to show progress
        for i in tqdm(range(0, total_frames, frame_interval), desc=f"Processing {video_path}"):
            
            ret, img = cap.read()
            if not ret:
                break

            frame_counter += 1

            try:
                # Run the YOLO model on the frame
                results = yolo_model(img, verbose=False, device=device)

                # Convert the detections to the required format: N X (x, y, x, y, conf, cls)
                dets = []
                for result in results:
                    for detection in result.boxes.data.cpu().numpy():
                        x1, y1, x2, y2, conf, cls = detection
                        dets.append([x1, y1, x2, y2, conf, int(cls)])
                dets = np.array(dets)
                if len(dets) == 0:
                    dets = np.empty((0, 6))

                # Update tracker with detections
                tracker.update(dets, img)

                if tracker.per_class_active_tracks:
                    for k in tracker.per_class_active_tracks.keys():
                        active_tracks = tracker.per_class_active_tracks[k]
                        for a in active_tracks:
                            if a.history_observations:
                                box = a.history_observations[-1]
                                
                                # Update tracking data
                                tracking_data[a.id]['frameid'].append(frame_counter)
                                tracking_data[a.id]['class'].append(a.cls)
                                tracking_data[a.id]['box'].append(box.tolist())
                                
                                center_x = int((box[0] + box[2]) / 2)
                                center_y = int((box[1] + box[3]) / 2)
                                tracking_data[a.id]['tracklets'].append([center_x, center_y])
                else:
                    for a in tracker.active_tracks:
                        if a.history_observations:
                            box = a.history_observations[-1]
                            
                            # Update tracking data
                            tracking_data[a.id]['frameid'].append(frame_counter)
                            tracking_data[a.id]['class'].append(a.cls)
                            tracking_data[a.id]['box'].append(box.tolist())
                            
                            center_x = int((box[0] + box[2]) / 2)
                            center_y = int((box[1] + box[3]) / 2)
                            tracking_data[a.id]['tracklets'].append([center_x, center_y])

            except Exception as e:
                print(f"An error occurred: {e}")
                break

        cap.release()

    # Convert defaultdict to regular dict for JSON serialization
    tracking_data = {k: dict(v) for k, v in tracking_data.items()}

    # At the end of the function
    tracking_json = os.path.join('runs', f'{cam_id}_tracking_data.json')
    with open(tracking_json, 'w') as f:
        json.dump(tracking_data, f)

    return tracking_json

# You would call the function like this:
# process_videos('input.json')