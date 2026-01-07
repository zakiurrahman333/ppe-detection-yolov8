# This script performs face recognition using the `face_recognition` library and provides real-time feedback.
# It uses OpenCV to process video streams or images and Pyttsx3 for text-to-speech announcements.
# The script includes:
# - Loading known faces from a directory and encoding them.
# - Recognizing faces in real-time from webcam, video files, or static images.
# - Providing voice feedback for recognized faces using a threaded text-to-speech engine.
# - Displaying annotated video frames or images with the recognized faces.

import face_recognition
import cv2
import os
import numpy as np
import pyttsx3
import threading
import queue

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Create a queue for managing text-to-speech requests
speech_queue = queue.Queue()

def speech_worker():
    """Worker thread to process speech requests sequentially."""
    while True:
        text = speech_queue.get()  # Block until an item is available
        if text is None:  # Stop the worker if None is received
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start the speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak(text):
    """Add text to the speech queue."""
    if text:
        speech_queue.put(text)

def load_known_faces(KNOWN_FACES_DIR):
    """Load and encode known faces from a directory."""
    if not os.path.exists(KNOWN_FACES_DIR):
        raise FileNotFoundError(f"The directory {KNOWN_FACES_DIR} does not exist.")
    
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])  # Use filename as the name
            else:
                print(f"Warning: No face detected in {filename}")

    if not known_face_encodings:
        raise ValueError("No valid faces found in the directory.")

    return known_face_encodings, known_face_names

def recognize_face(face_encoding, known_face_encodings, known_face_names, last_name):
    """Recognize the face and provide feedback."""
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # Use the best match if available
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    if matches and matches[np.argmin(face_distances)]:
        best_match_index = np.argmin(face_distances)
        name = known_face_names[best_match_index]

        # Provide voice feedback only if the name has changed
        if name != last_name[0]:
            speak(f"Hello, {name}!")
            last_name[0] = name
    else:
        if name != last_name[0]:
            speak("Unknown person detected")
            last_name[0] = name

    return name

def process_video(KNOWN_FACES_DIR, input_source=0, frame_rate=1):
    """Start face recognition for webcam, video, or image input."""
    try:
        # Load known faces
        known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)
        
        # Determine input source type (0 for webcam, string for file path)
        if isinstance(input_source, str) and os.path.isfile(input_source):
            if input_source.lower().endswith((".jpg", ".jpeg", ".png")):
                process_image(input_source, known_face_encodings, known_face_names)
                return
            elif input_source.lower().endswith((".mp4", ".avi", ".mov")):
                video_capture = cv2.VideoCapture(input_source)
            else:
                raise ValueError("Unsupported file format. Only image or video files are allowed.")
        else:
            video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            raise RuntimeError("Could not access the input source.")

        cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
        print("Starting video capture...")

        frame_counter = 0
        last_name = [None]  # To store the last recognized name (mutable for shared access)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces and their encodings every few frames
            if frame_counter % frame_rate == 0:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                # Process each detected face
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    name = recognize_face(face_encoding, known_face_encodings, known_face_names, last_name)

                    # Scale face location back to the original frame size
                    top, right, bottom, left = [v * 4 for v in face_location]
                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    # Label the face with the name
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            frame_counter += 1

            # Display the annotated frame
            cv2.imshow("Face Recognition", frame)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Release video resources and close OpenCV windows
        if 'video_capture' in locals() and video_capture.isOpened():
            video_capture.release()
        cv2.destroyAllWindows()
        print("Video capture stopped.")

        # Stop the speech thread
        speech_queue.put(None)  # Signal the worker thread to stop
        speech_thread.join()

def process_image(image_path, known_face_encodings, known_face_names):
    """Process a static image for face recognition."""
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if not face_encodings:
            print("No faces found in the image.")
            return

        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = recognize_face(face_encoding, known_face_encodings, known_face_names, [None])

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Convert to BGR for OpenCV compatibility and display
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Recognition - Image", bgr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    KNOWN_FACES_DIR = "images/known_faces"
    input_source = input("Enter input source (0 for webcam, or provide path to image/video): ")
    process_video(KNOWN_FACES_DIR, input_source)
