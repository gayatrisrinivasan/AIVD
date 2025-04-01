import cv2
import torch
import numpy as np
import os

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Define classes for detection
PERSON_CLASS = 0  # YOLOv5 class for 'person'
MOTORCYCLE_CLASS = 3  # YOLOv5 class for 'motorcycle'

def detect_violations(image_path):
    img = cv2.imread(image_path)

    # Run YOLOv5 inference
    results = model(img)
    
    persons = []
    motorcycles = []

    # Process detections
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        class_id = int(cls)

        if class_id == PERSON_CLASS:
            persons.append((x1, y1, x2, y2))
        elif class_id == MOTORCYCLE_CLASS:
            motorcycles.append((x1, y1, x2, y2))

    # Overloading Violation Check (More than 2 persons on a motorcycle)
    overloading_violation = len(persons) > 2

    # Draw bounding boxes
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for person
    for (x1, y1, x2, y2) in motorcycles:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for motorcycle

    # Display results
    if overloading_violation:
        print("Overloading Violation Detected!")
        cv2.putText(img, "Overloading!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Ensure the violations folder exists
    output_dir = "violations"
    os.makedirs(output_dir, exist_ok=True)

    # Save the processed image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)

    print(f"Saved result to {output_path}")

# Example usage
image_path = "uploads/example.jpg"  # Change this to your uploaded image
detect_violations(image_path)