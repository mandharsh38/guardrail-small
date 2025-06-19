import os
import json
from PIL import Image

# Only keep and remap these labels
LABEL_REMAP = {
    "DamagedCrashBarrier": "DamagedCrashBarrier",
    "VegetationCrashBarrier": "DamagedCrashBarrier",
    "CrashBarrier": "CrashBarrier"
}

# Final class IDs after remap
FINAL_LABELS = {
    "CrashBarrier": 0,
    "DamagedCrashBarrier": 1
}

def convert_filtered_labelme_to_yolov8_seg(source_folder, output_label_folder):
    os.makedirs(output_label_folder, exist_ok=True)

    for file in os.listdir(source_folder):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(source_folder, file)
        image_filename = os.path.splitext(file)[0] + ".jpg"
        image_path = os.path.join(source_folder, image_filename)

        if not os.path.exists(image_path):
            print(f"⚠️ Image not found for: {file}")
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        with open(json_path, 'r') as f:
            data = json.load(f)

        yolo_lines = []

        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label not in LABEL_REMAP:
                continue  # Ignore unwanted labels

            mapped_label = LABEL_REMAP[label]
            class_id = FINAL_LABELS[mapped_label]

            points = shape.get("points", [])
            if len(points) < 3:
                continue  # Skip too-small polygons

            # Flatten and normalize polygon points
            flat_points = []
            for x, y in points:
                flat_points.append(str(round(x / img_w, 6)))
                flat_points.append(str(round(y / img_h, 6)))

            yolo_lines.append(f"{class_id} " + " ".join(flat_points))

        # Save only if any valid labels remain
        if yolo_lines:
            out_txt = os.path.join(output_label_folder, os.path.splitext(image_filename)[0] + ".txt")
            with open(out_txt, "w") as out_f:
                out_f.write("\n".join(yolo_lines))
            print(f"✅ Converted: {file}")
        else:
            print(f"🗑️ Skipped: {file} (no valid labels)")

# === USAGE ===
SOURCE_FOLDER = "frames/"
OUTPUT_LABEL_FOLDER = "dataset/labels/train/"

convert_filtered_labelme_to_yolov8_seg(SOURCE_FOLDER, OUTPUT_LABEL_FOLDER)
