"""This code leverages the YOLO model to detect and highlight humans in video files, 
and optionally saves the processed videos to an output directory"""
# This is further integrated in the ppe_detection_final.py script


import cv2
import numpy as np
import os

# Constants
CONF_THRESHOLD = 0.5  # Confidence threshold
NMS_THRESHOLD = 0.4   # Non-Maximum Suppression threshold
FRAME_SKIP = 1        # Skip every nth frame for faster processing

# Load YOLO model
def load_yolo(weights_path, config_path, classes_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(classes_path, "r") as file:
        classes = [line.strip() for line in file.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Process a single video
def process_video(video_path, net, classes, output_layers, output_dir=None):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer if output directory is provided
    video_writer = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"processed_{os.path.basename(video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for faster processing
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        height, width, _ = frame.shape

        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process YOLO outputs
        boxes, confidences, class_ids = [], [], []
        for out_layer in outs:
            for detection in out_layer:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Consider only 'person' detections with high confidence
                if confidence > CONF_THRESHOLD and class_id == classes.index("person"):
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

        # Draw bounding boxes
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show frame
        cv2.imshow("Human Detection", frame)

        # Write frame to output video if enabled
        if video_writer is not None:
            video_writer.write(frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    print(f"Finished processing {video_path}")


# Main
def main():
    # Paths to YOLO model files
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    classes_path = "coco.names"

    # Load YOLO model
    net, classes, output_layers = load_yolo(weights_path, config_path, classes_path)

    # Folder containing videos
    video_folder = "videos"
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

    # Output directory for processed videos
    output_dir = "processed_videos"

    # Process each video
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        process_video(video_path, net, classes, output_layers, output_dir)

if __name__ == "__main__":
    main()
