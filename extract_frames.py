import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=5):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_rate == 0:
            frame_filename = f"{output_folder}/frame_{count}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
            count += 1
        frame_id += 1

    cap.release()
    print("Frame extraction completed.")