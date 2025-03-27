from violation_detection import detect_violations
import os

input_folder = "roi_results"

os.makedirs("violations", exist_ok=True)

# Run detection on all images
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        detect_violations(os.path.join(input_folder, filename))