import cv2
import torch
import argparse
import os

# Load the YOLOv5s pretrained model (COCO dataset)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Define class IDs (from COCO, available in YOLOv5s)
PERSON_CLASS = 0      # "person" in COCO
MOTORCYCLE_CLASS = 3  # "motorcycle" in COCO

# Create output folder if it doesn't exist
os.makedirs("results", exist_ok=True)

def detect_violations(image_path, conf_thresh=0.5):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image", image_path)
        return

    # Run inference on the image
    results = model(img)  # No 'conf' keyword here

    persons = []
    motorcycles = []
    
    # Process detections manually with confidence filtering
    for *xyxy, conf, cls in results.xyxy[0]:
        if float(conf) < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        class_id = int(cls)
        if class_id == PERSON_CLASS:
            persons.append((x1, y1, x2, y2))
        elif class_id == MOTORCYCLE_CLASS:
            motorcycles.append((x1, y1, x2, y2))
    
    # Initialize violation flags
    helmet_violation = False
    overloading_violation = False

    # Heuristic for helmet violation:
    # Since the COCO model doesn't detect helmets, we assume that
    # if a motorcycle is detected with at least one person riding, that person has no helmet.
    for moto in motorcycles:
        mx1, my1, mx2, my2 = moto
        riders = [p for p in persons if (mx1 < (p[0] + p[2]) // 2 < mx2 and my1 < (p[1] + p[3]) // 2 < my2)]
        if riders:
            helmet_violation = True
            for p in riders:
                cv2.putText(img, "NO HELMET", (p[0], p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Overloading violation: if more than 2 persons are detected riding on any motorcycle
    for moto in motorcycles:
        mx1, my1, mx2, my2 = moto
        riders = [p for p in persons if (mx1 < (p[0] + p[2]) // 2 < mx2 and my1 < (p[1] + p[3]) // 2 < my2)]
        if len(riders) > 2:
            overloading_violation = True
            cv2.putText(img, "OVERLOADING", (mx1, my1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # If no violation detected, display a message
    if not helmet_violation and not overloading_violation:
        cv2.putText(img, "No Violation Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw bounding boxes for visualization
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for person
    for (x1, y1, x2, y2) in motorcycles:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for motorcycle

    # Save the output image in the 'results' folder
    output_path = os.path.join("results", os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print("Result saved to:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect traffic violations from an image using a pre-trained model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file (e.g., uploads/example.jpg)")
    args = parser.parse_args()
    detect_violations(args.image)