import cv2
import torch
import argparse
import os

# YOLOv5s pretrained - COCO dataset
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

PERSON_CLASS = 0      # person
MOTORCYCLE_CLASS = 3  # motorcycle

os.makedirs("results", exist_ok=True)

def detect_violations(image_path, conf_thresh=0.5):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image", image_path)
        return

    results = model(img)

    persons = []
    motorcycles = []
    
    for *xyxy, conf, cls in results.xyxy[0]:
        if float(conf) < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        class_id = int(cls)
        if class_id == PERSON_CLASS:
            persons.append((x1, y1, x2, y2))
        elif class_id == MOTORCYCLE_CLASS:
            motorcycles.append((x1, y1, x2, y2))
    
    helmet_violation = False
    overloading_violation = False

    for moto in motorcycles:
        mx1, my1, mx2, my2 = moto
        riders = [p for p in persons if (mx1 < (p[0] + p[2]) // 2 < mx2 and my1 < (p[1] + p[3]) // 2 < my2)]
        if riders:
            helmet_violation = True
            for p in riders:
                cv2.putText(img, "NO HELMET", (p[0], p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    for moto in motorcycles:
        mx1, my1, mx2, my2 = moto
        riders = [p for p in persons if (mx1 < (p[0] + p[2]) // 2 < mx2 and my1 < (p[1] + p[3]) // 2 < my2)]
        if len(riders) > 2:
            overloading_violation = True
            cv2.putText(img, "OVERLOADING", (mx1, my1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if not helmet_violation and not overloading_violation:
        cv2.putText(img, "No Violation Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for person
    for (x1, y1, x2, y2) in motorcycles:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for motorcycle

    output_path = os.path.join("results", os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print("Result saved to:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect traffic violations from an image using a pre-trained model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file (e.g., uploads/example.jpg)")
    args = parser.parse_args()
    detect_violations(args.image)
