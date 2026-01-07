# This script is designed for detecting Personal Protective Equipment (PPE) compliance using YOLO models.
# It can process images, video files, or live webcam feeds to detect whether individuals are wearing necessary PPE items such as hardhats, vests, gloves, and boots.
# The script includes the following functionalities:
# - Initialize and load the YOLO models for PPE and Worker detection.
# - Process input sources (image, video, webcam) to detect PPE compliance.
# - Play a buzzer sound if non-compliance is detected (e.g., missing hardhat, vest, gloves, or boots).
# - Use OpenCV to display annotated images or video frames with detected items.
# - Utilize threading for non-blocking buzzer playback.
# - Log detailed information and errors for debugging purposes.

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import pygame  # For buzzer sound
import os
import time
import threading
import logging

# Define paths for models and buzzer audio
PPE_MODEL_PATH = "model/yolo.pt"
WORKER_MODEL_PATH = "model/person.pt"
BUZZER_AUDIO_PATH = "buzzer.wav"

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)

# Initialize buzzer
def initialize_buzzer():
    if os.path.exists(BUZZER_AUDIO_PATH):
        pygame.mixer.init()
        pygame.mixer.music.load(BUZZER_AUDIO_PATH)
        logging.info("Buzzer audio loaded successfully.")
    else:
        logging.error("Buzzer audio file not found!")

# Play buzzer sound
def play_buzzer():
    if pygame.mixer.get_init():
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.delay(10)  # Non-blocking wait
    else:
        logging.error("Buzzer not initialized. Skipping buzzer sound.")

# Load YOLO model
def load_model(model_path):
    try:
        logging.info(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        logging.info(f"Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        exit()

# Detect PPE compliance in an image
def detect_ppe_in_image(image_path, model, confidence):
    try:
        logging.info(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to load image.")
        
        results = model.predict(image, conf=confidence)
        res_image = results[0].plot()  # Annotated image

        # Check detections and trigger buzzer
        buzzer_triggered = False
        detected_items = []

        for box in results[0].boxes:
            class_name = model.names[int(box.cls)]
            confidence_score = float(box.conf)  # Ensure this is a float
            detected_items.append(f"{class_name} (Confidence: {confidence_score:.2f})")

            if class_name in ["no_vest", "no_hardhat", "no_gloves"]:
                buzzer_triggered = True

        # Print the detected items in the terminal
        print("\nDetected Items:")
        for item in detected_items:
            print(item)

        if buzzer_triggered:
            logging.warning("Non-compliance detected! Playing buzzer...")
            play_buzzer()

        # Display result
        cv2.imshow("Detected PPE", res_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"Error detecting PPE in image: {e}")

# Function to play buzzer sound in a separate thread
def play_buzzer_async():
    if pygame.mixer.get_init():
        pygame.mixer.music.play()
    else:
        logging.error("Buzzer not initialized. Skipping buzzer sound.")

# Detect PPE compliance in a video
def detect_ppe_in_video(video_path, model, confidence):
    try:
        print(model.names)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video file.")

        # Get the video frame rate (FPS)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"INFO:root:Video FPS: {fps}")  # More detailed info

        # Create the OpenCV window and make it stay on top
        cv2.namedWindow("Detected PPE in Video", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Detected PPE in Video", cv2.WND_PROP_TOPMOST, 1)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Run model inference
            results = model.predict(frame, conf=confidence, verbose=False)
            
            # Annotated frame (bounding boxes)
            res_frame = results[0].plot()

            # Check detections and trigger buzzer if needed
            buzzer_triggered = False
            detected_items = []

            for box in results[0].boxes:
                class_name = model.names[int(box.cls)]
                confidence_score = float(box.conf)  # Ensure this is a float
                detected_items.append(f"{class_name} (Confidence: {confidence_score:.2f})")
                
                # Print detected items
                if class_name in ["no_vest", "no_hardhat", "no_gloves", "no_boots"]:
                    buzzer_triggered = True

            # Print the detected items in the terminal
            print("\nDetected Items in Current Frame:")
            for item in detected_items:
                print(item)

            if buzzer_triggered:
                print("\nNon-compliance detected in video frame! Playing buzzer...")
                threading.Thread(target=play_buzzer_async).start()

            # Display the frame
            cv2.imshow("Detected PPE in Video", cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR))

            # Control the frame rate to match the video FPS
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                break  # Exit on 'q' key press

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"ERROR:root:Error detecting PPE in video: {e}")

# Detect PPE compliance using webcam
def detect_ppe_using_webcam(model, confidence):
    try:
        cap = cv2.VideoCapture(0)  # Use default webcam
        if not cap.isOpened():
            raise ValueError("Unable to open webcam.")

        # Create the OpenCV window and make it stay on top
        cv2.namedWindow("Detected PPE from Webcam", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Detected PPE from Webcam", cv2.WND_PROP_TOPMOST, 1)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit if no frame is captured

            # Run model inference
            results = model.predict(frame, conf=confidence, verbose=False)

            # Annotated frame (bounding boxes)
            res_frame = results[0].plot()

            # Check detections and trigger buzzer if needed
            buzzer_triggered = False
            detected_items = []

            for box in results[0].boxes:
                class_name = model.names[int(box.cls)]
                confidence_score = float(box.conf)  # Ensure this is a float
                detected_items.append(f"{class_name} (Confidence: {confidence_score:.2f})")

                # Print detected items
                if class_name in ["no_vest", "no_hardhat", "no_gloves", "no_boots"]:
                    buzzer_triggered = True

            # Print the detected items in the terminal
            print("\nDetected Items in Current Frame:")
            for item in detected_items:
                print(item)

            if buzzer_triggered:
                print("\nNon-compliance detected in webcam feed! Playing buzzer...")
                threading.Thread(target=play_buzzer_async).start()

            # Display the frame
            cv2.imshow("Detected PPE from Webcam", cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR))

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"ERROR:root:Error detecting PPE in webcam feed: {e}")

# Main function
def main():
    pygame.init()  # Initialize pygame for buzzer
    initialize_buzzer()

    # Select model
    model_type = input("Select Model (PPE/Worker): ").strip().lower()
    if model_type == "ppe":
        model_path = PPE_MODEL_PATH
    elif model_type == "worker":
        model_path = WORKER_MODEL_PATH
    else:
        logging.error("Invalid model type selected. Exiting.")
        return

    model = load_model(model_path)

    # Select source
    source_type = input("Select Source (Image/Video/Webcam): ").strip().lower()
    if source_type == "image":
        image_path = input("Enter path to the image file: ").strip()
        if Path(image_path).is_file():
            detect_ppe_in_image(image_path, model, CONFIDENCE_THRESHOLD)
        else:
            logging.error("Invalid image path. Exiting.")

    elif source_type == "video":
        video_path = input("Enter path to the video file: ").strip()
        if Path(video_path).is_file():
            detect_ppe_in_video(video_path, model, CONFIDENCE_THRESHOLD)
        else:
            logging.error("Invalid video path. Exiting.")

    elif source_type == "webcam":
        detect_ppe_using_webcam(model, CONFIDENCE_THRESHOLD)

    else:
        logging.error("Invalid source type. Exiting.")

if __name__ == "__main__":
    main()
