import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Define classes for detection
PERSON_CLASS = 0  # YOLOv5 class for 'person'
HELMET_CLASS = 26  # Custom trained model for 'helmet'
MOTORCYCLE_CLASS = 3  # YOLOv5 class for 'motorcycle'

def detect_violations(image_path):
    img = cv2.imread(image_path)

    # Run YOLOv5 inference
    results = model(img)
    
    persons = []
    helmets = []
    motorcycles = []

    # Process detections
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        class_id = int(cls)

        if class_id == PERSON_CLASS:
            persons.append((x1, y1, x2, y2))
        elif class_id == HELMET_CLASS:
            helmets.append((x1, y1, x2, y2))
        elif class_id == MOTORCYCLE_CLASS:
            motorcycles.append((x1, y1, x2, y2))

    # Helmet Violation Check
    helmet_violation = False
    for person in persons:
        x1_p, y1_p, x2_p, y2_p = person
        has_helmet = any(x1_h < x1_p and x2_h > x2_p for x1_h, y1_h, x2_h, y2_h in helmets)
        if not has_helmet:
            helmet_violation = True
            cv2.putText(img, "NO HELMET!", (x1_p, y1_p - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Overloading Violation Check
    overloading_violation = len(persons) > 2  # More than 2 persons on a motorcycle

    # Wrong-Way Detection (Basic implementation)
    wrong_way_violation = False  # Future Implementation - Analyze vehicle direction

    # Draw bounding boxes
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for person
    for (x1, y1, x2, y2) in helmets:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for helmet
    for (x1, y1, x2, y2) in motorcycles:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for motorcycle

    # Display results
    if helmet_violation:
        print("Helmet Violation Detected!")
    if overloading_violation:
        print("Overloading Violation Detected!")
    if wrong_way_violation:
        print("Wrong-Way Violation Detected!")

    # Save result
    output_path = "violations/" + image_path.split("/")[-1]
    cv2.imwrite(output_path, img)
    print(f"Saved result to {output_path}")