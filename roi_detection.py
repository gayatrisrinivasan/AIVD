import torch
import cv2
import os
import numpy as np

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

def detect_roi(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            results = model(img)

            for *xyxy, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img)
            print(f"ROI detected: {output_path}")

    print("ROI Detection Completed.")
