import cv2
import os
import numpy as np

def preprocess_frames(input_folder, output_folder, resize_dim=(640, 480)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            # Noise Reduction using Gaussian Blur
            denoised = cv2.GaussianBlur(img, (5, 5), 0)

            # Resizing
            resized = cv2.resize(denoised, resize_dim)

            # Save the processed frame
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized)
            print(f"Processed: {output_path}")

    print("Preprocessing completed.")