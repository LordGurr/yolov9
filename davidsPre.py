print("Starting Project")
# basic imports
import os
import sys
import yaml

# Preprocessing
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import defaultdict

# network imports
#import yolov9.train
#from ultralytics import YOLO as yolov9
import torch


# logging imports
#import wandb
if torch.cuda.is_available():
    device = "cuda" #Nvidia Graphics Card
elif torch.backends.mps.is_available():
    device = "mps" # Apple
else:
    device = "cpu" # Worst Case
print(device)

xs = []
ys_train = []
ys_valid = []

Paths = [("Train/2022-12-04 Bjenberg 02/2022-12-04 Bjenberg 02.MP4", "frames/2022-12-04 Bjenberg 02"),
         ("Train/2022-12-02 Asjo 01_stabilized/2022-12-02 Asjo 01_stabilized.MP4", "frames/2022-12-02 Asjo 01_stabilized"),
         ("Train/2022-12-23 Asjo 01_HD 5x stab/2022-12-23 Asjo 01_HD 5x stab.MP4", "frames/2022-12-23 Asjo 01_HD 5x stab")
          ]

for video_path, output_dir in Paths:

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path) # Choose which file and open video reading

    if not cap.isOpened(): # If it cannot see the video
        raise ValueError

    framed_id = 0

    while True:
        ret, frame = cap.read()
        if not ret: # ret is boolean for how it did go to read the frame from the video.
            break

        frame_path = os.path.join(output_dir, f"frame_{framed_id:06d}.jpg")
        cv2.imwrite(frame_path, frame)

        framed_id += 1
    cap.release() # Closes the file, like file.close but for videos and cameras.

print(f"Saved {framed_id} at the location {output_dir} <-- Last folder that was saved")

img = cv2.imread("frames/2022-12-04 Bjenberg 02/frame_000333.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis("off")
plt.show()

# ----------------------------------- Load and Structure Annotations form XML files -----------------------------------

xml_files = ["Train/2022-12-04 Bjenberg 02/2022-12-04 Bjenberg 02.xml",
             "Train/2022-12-02 Asjo 01_stabilized/2022-12-02 Asjo 01_stabilized.xml",
             "Train/2022-12-23 Asjo 01_HD 5x stab/2022-12-23 Asjo 01_HD 5x stab.xml"
             ]

# Creates a Cache memory from all sequences
all_frame_annotations = {}

for xml_path in xml_files:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []

    for track in root.findall("track"):
        track_id = int(track.attrib["id"])
        label = track.attrib["label"]

        for box in track.findall("box"):
            frame = int(box.attrib["frame"])
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            annotations.append({
                "track_id": track_id,
                "label": label,
                "frame": frame,
                "bbox": [xtl, ytl, xbr, ybr]
            })

    frame_annotations = defaultdict(list)

    for ann in annotations:
        frame_annotations[ann["frame"]].append({
            "track_id": ann["track_id"],
            "label": ann["label"],
            "bbox": ann["bbox"]
        })

    folder_name = os.path.basename(os.path.dirname(xml_path))
    all_frame_annotations[folder_name] = frame_annotations

    print("Antal frames med annotationer:", len(frame_annotations))
    print("Exempel frame 0:", frame_annotations[0])

# ----------------------------------- Visualization -----------------------------------

video_path = "Train/2022-12-04 Bjenberg 02/2022-12-04 Bjenberg 02.MP4"
target_frame = 200

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame) # Start from index values target_frame
ret, frame = cap.read()
cap.release() # We only read one file, therefore we close this very quickly


if not ret: 
    raise ValueError

folder_name = "2022-12-04 Bjenberg 02"
for ann in all_frame_annotations[folder_name][target_frame]: # all frames with annotations
    x1, y1, x2, y2 = map(int, ann["bbox"]) # Get box-coordinates
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText( # write Car over box
        frame,
        ann["label"],
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0), # Green Colour
        2
    )

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(frame_rgb)
plt.axis("off")
plt.show()

# ----------------------------------- Save Annotations in YOLO Format -----------------------------------

frames_base_dir = "frames"
labels_base_dir = "labels"

os.makedirs(labels_base_dir, exist_ok=True)

class_map = { # Car is labelled zero
    "car": 0
}

for folder_name, frame_annotations in all_frame_annotations.items():

    frame_folder = os.path.join(frames_base_dir, folder_name)
    label_folder = os.path.join(labels_base_dir, folder_name)

    os.makedirs(label_folder, exist_ok=True)

    for frame_id, anns in frame_annotations.items():
        image_path = os.path.join(frame_folder, f"frame_{frame_id:06d}.jpg")

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        img_h, img_w = image.shape[:2]

        label_path = os.path.join(label_folder, f"frame_{frame_id:06d}.txt")

        with open(label_path, "w") as f:
            for ann in anns:
                label = ann["label"]

                if label not in class_map:
                    continue

                x1, y1, x2, y2 = ann["bbox"]

                # Keep coordinates inside image boundaries
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                class_id = class_map[label]

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")