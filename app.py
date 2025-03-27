from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
from violation_detection import detect_violations

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "violations"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    """ Home page """
    files = os.listdir(RESULT_FOLDER)
    return render_template("index.html", files=files)

@app.route("/upload", methods=["POST"])
def upload_file():
    """ Handle image uploads """
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Run violation detection
    detect_violations(filepath)

    return "File processed successfully", 200

@app.route("/violations/<filename>")
def get_violation_image(filename):
    """ Serve detected violation images """
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)