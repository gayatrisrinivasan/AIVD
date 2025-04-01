# AI-Powered Traffic Violation Detection

### Install required libraries
```bash
pip install -r requirements.txt
```
---
## How to Run

### 1. Extract Frames from Video
```bash
python extract_frames.py
```

### 2. Preprocess Frames
```bash
python preprocess.py
```

### 3. Detect Region of Interest
```bash
python roi_detection.py
```

### 4. Run Violation Detection on an Image
```bash
python violation_detection.py --image uploads/example.jpg
```

## What Violations are Detected?
- **Helmet Violation** → Detects if a rider is not wearing a helmet.
- **Overloading Violation** → Detects if more than two people are on a motorcycle.
- **Wrong-Way Violation** → Future implementation (detects vehicles moving in the wrong direction).
