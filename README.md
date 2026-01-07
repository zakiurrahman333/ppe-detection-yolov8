# PPE Detection System

## Overview

This project implements a **PPE Detection System**, **Human Detection using YOLOv3**, and **Face Recognition**. These systems can analyze images, videos, or webcam feeds to detect compliance with safety requirements, human presence, and recognize faces. Alerts are triggered when violations are detected, such as non-compliance with PPE or an unknown face.

## Features

- **Image Detection**: Analyze images for PPE compliance.
- **Video Detection**: Process video files to detect non-compliance.
- **Webcam Detection**: Monitor a live webcam feed for PPE violations.
- **Audio Alert**: Plays a buzzer sound when non-compliance is detected.
- **Real-Time Display**: Annotated outputs with detected objects are displayed.

## Prerequisites

Ensure you have **Python 3.8+** installed.

### Dependencies

Install the required libraries using the following commands:

```bash
pip install ultralytics opencv-python numpy pygame
pip install opencv-python-headless  # For YOLOv3 Human Detection
```

## Files Needed

### PPE Detection Model:
- **yolo.pt**: Trained YOLO model for PPE detection (helmets, vests, gloves).
- **person.pt**: Model for human detection.

### Human Detection Model:
- **yolov3.weights**: YOLOv3 pre-trained weights.
- **yolov3.cfg**: YOLOv3 configuration file.
- **coco.names**: Class names for YOLOv3.

### Face Recognition:
- **Known Faces**: Folder containing images of known individuals for face recognition (e.g., `person1.jpg`, `person2.jpg`).
- **Cascade Classifier**: Pre-trained `haarcascade_frontalface_default.xml` for face detection.

### Audio Alert:
- **buzzer.wav**: The buzzer sound to trigger on violation detection.

## Usage

### 1. Human Detection

To run human detection:

```bash
python human_detection.py
```

#### Prepare Video Files

- Place your video files in the `videos` directory.
- Supported formats: `.mp4`, `.avi`.

#### Processing

- The script processes all videos in the `videos` folder.
- It detects humans in each video and saves the processed videos in the `processed_videos` folder.


### 2. PPE Detection

To run the PPE detection system:

```bash
python ppe_detection_final.py
```

#### Select Model:

- PPE Model: Detects PPE (helmets, vests, gloves).
- Worker Model: Detects workers in restricted zones.

#### Select Source:

- Image: Provide the path to an image file.
- Video: Provide the path to a video file.
- Webcam: Use the default webcam for real-time monitoring.

### 3. Face Recognition

To run face recognition:

```bash
python face_recognition1.py
```

#### Select Input Type:

- Image: Provide the path to an image file.
- Video: Provide the path to a video file.
- Webcam: Use the default webcam for real-time face recognition.

#### Known Faces Folder:

Place images of known individuals in the known_faces folder (e.g., person1.jpg, person2.jpg).

## Key Functions

#### PPE Detection:

- Buzzer Alert: Plays a buzzer sound when PPE violations are detected.
- YOLOv8 Model: Dynamically loads the YOLOv8 model to detect PPE and workers.
- Real-Time Display: Shows annotated output for detected PPE.

#### Human Detection (YOLOv3):

- Human Detection: Identifies humans in video frames.
- Processed Videos: Annotated videos are saved in the processed_videos folder.

#### Face Recognition:

- Face Identification: Matches faces in images or videos with known individuals.
- Real-Time Webcam Feed: Supports live webcam recognition.


## Configuration

Modify the neceassry paths in the scripts as per your system.


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/AyushMaurya1/Ppe_Kit_Detection/blob/main/LICENSE.txt) file for details.

