import os
import cv2
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Detect and save frames with damaged objects (no boxes).")
parser.add_argument("video_path", type=str, help="Path to input video file")
parser.add_argument("--output_folder", type=str, default="damaged_frames", help="Folder to save detected frames")
parser.add_argument("--skip_frames", type=int, default=30, help="Number of frames to skip after detection")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection")
args = parser.parse_args()

model = YOLO("runs/segment/train3/weights/best.pt")

os.makedirs(args.output_folder, exist_ok=True)
cap = cv2.VideoCapture(args.video_path)

frame_count = 0
saved_count = 0
skip_until = -1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count < skip_until:
        frame_count += 1
        continue

    result = model.predict(source=frame, conf=args.conf, verbose=False)[0]
    found_damaged = any(int(box.cls.item()) == 1 for box in result.boxes)

    if found_damaged:
        out_path = os.path.join(args.output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(out_path, frame)
        saved_count += 1
        skip_until = frame_count + args.skip_frames

    frame_count += 1
    print(f"\r{frame_count} frames processed, saved {saved_count} damaged frames", end="", flush=True)

cap.release()
print(f"\nDone. Saved {saved_count} frames to: {args.output_folder}")
