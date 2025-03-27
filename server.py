from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Load YOLOv5 model
model = YOLO("best.pt")  # Replace with your trained model path

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Run YOLO detection
    results = model(filepath)
    violations = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = round(float(box.conf[0]) * 100, 1)
            
            # Define violation labels based on your model's classes
            class_labels = {0: "No Helmet", 1: "Overloading", 2: "Use of phone while driving"}
            violation_type = class_labels.get(class_id, "Unknown Violation")

            violations.append({
                "type": violation_type,
                "confidence": f"{confidence}%",
                "location": "Detected on Road"
            })

    response = {
        "violations": violations,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)