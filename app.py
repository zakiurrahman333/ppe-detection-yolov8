import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import pandas as pd
from datetime import datetime
import pygame
import os

# Load custom YOLOv8 PPE model
model = YOLO("yolov8n.pt")  # <-- Replace with your model path

# Sidebar Controls
st.sidebar.title("⚙️ Control Panel")
alert_on = st.sidebar.toggle("🔔 Buzzer Alert", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Detection Stats")
violation_count_display = st.sidebar.empty()
helmet_display = st.sidebar.empty()
vest_display = st.sidebar.empty()
gloves_display = st.sidebar.empty()
boots_display = st.sidebar.empty()

# Load Buzzer
try:
    pygame.mixer.init()
    pygame.mixer.music.load("buzzer.wav")
    buzzer_ready = True
except:
    buzzer_ready = False

# PPE Classes
PPE_CLASSES = ["helmet", "vest", "gloves", "boots", "no_helmet", "no_vest", "no_gloves", "no_boots", "person"]

# App Title
st.title("🦺 PPE Detection System (YOLOv8 + Streamlit)")

# Input Source Selector
input_type = st.selectbox("Select Input Source", ["Image", "Video", "Webcam"])

# Global Detection Log
detection_log = []

# Class Counting Function
def count_ppe(results):
    names = results.names
    detected_classes = [names[int(cls)] for cls in results.boxes.cls.cpu().numpy()]
    count = {cls: detected_classes.count(cls) for cls in PPE_CLASSES if cls in detected_classes}
    return count

# Play Buzzer Function
def play_buzzer():
    if buzzer_ready:
        pygame.mixer.music.play()

# Log Detections + Auto Snapshot
def log_detections(results, frame=None):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    snapshot_taken = False
    saved_image = ""

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        confidence = float(box.conf[0])

        if label.startswith("no_") and frame is not None and not snapshot_taken:
            os.makedirs("snapshots", exist_ok=True)
            image_name = f"{now}.jpg"
            snapshot_path = os.path.join("snapshots", image_name)
            cv2.imwrite(snapshot_path, frame)
            saved_image = image_name
            snapshot_taken = True

        detection_log.append({
            "Timestamp": now,
            "Label": label,
            "Confidence": round(confidence, 2),
            "Image": saved_image if label.startswith("no_") else ""
        })

# Handle Image Upload
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        st.image(image_np, caption="Uploaded Image", use_column_width=True)

        results = model(image_bgr)[0]
        annotated = results.plot()
        st.image(annotated, caption="Detection Result", use_column_width=True)

        stats = count_ppe(results)
        log_detections(results, image_bgr)

        violations = sum(stats.get(cls, 0) for cls in ["no_helmet", "no_vest", "no_gloves", "no_boots"])
        if alert_on and violations > 0:
            play_buzzer()

        violation_count_display.metric("🚨 Violations", violations)
        helmet_display.metric("🪖 Helmets", stats.get("helmet", 0))
        vest_display.metric("🦺 Vests", stats.get("vest", 0))
        gloves_display.metric("🧤 Gloves", stats.get("gloves", 0))
        boots_display.metric("🥾 Boots", stats.get("boots", 0))

        if detection_log:
            df = pd.DataFrame(detection_log)
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📅 Download Violation Report as CSV", csv, "violation_report.csv", "text/csv")

# Handle Video Upload
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        total_violations = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)[0]
            annotated_frame = results.plot()

            stats = count_ppe(results)
            log_detections(results, frame)

            violations = sum(stats.get(cls, 0) for cls in ["no_helmet", "no_vest", "no_gloves", "no_boots"])
            total_violations += violations

            if alert_on and violations > 0:
                play_buzzer()

            violation_count_display.metric("🚨 Violations", total_violations)
            helmet_display.metric("🪖 Helmets", stats.get("helmet", 0))
            vest_display.metric("🦺 Vests", stats.get("vest", 0))
            gloves_display.metric("🧤 Gloves", stats.get("gloves", 0))
            boots_display.metric("🥾 Boots", stats.get("boots", 0))

            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()

        if detection_log:
            df = pd.DataFrame(detection_log)
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📅 Download Violation Report as CSV", csv, "violation_report.csv", "text/csv")

# Handle Webcam
elif input_type == "Webcam":
    camera_index = st.number_input("Enter Camera Index (e.g., 0 for built-in, 1 for USB)", min_value=0, max_value=10, value=0, step=1)

    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(camera_index)
        stframe = st.empty()

        if not cap.isOpened():
            st.error(f"❌ Unable to open camera index {camera_index}. Try a different number.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)[0]
                annotated_frame = results.plot()

                stats = count_ppe(results)
                log_detections(results, frame)

                violations = sum(stats.get(cls, 0) for cls in ["no_helmet", "no_vest", "no_gloves", "no_boots"])
                if alert_on and violations > 0:
                    play_buzzer()

                violation_count_display.metric("🚨 Violations", violations)
                helmet_display.metric("🪖 Helmets", stats.get("helmet", 0))
                vest_display.metric("🦺 Vests", stats.get("vest", 0))
                gloves_display.metric("🧤 Gloves", stats.get("gloves", 0))
                boots_display.metric("🥾 Boots", stats.get("boots", 0))

                stframe.image(annotated_frame, channels="BGR", use_column_width=True)

            cap.release()

            if detection_log:
                df = pd.DataFrame(detection_log)
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📅 Download Violation Report as CSV", csv, "violation_report.csv", "text/csv")
