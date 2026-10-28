import streamlit as st

st.set_page_config(page_title="Object Detection", layout="wide")

st.title("🎯 Real-Time Object Detection")
st.write("YOLOv8 Object Detection Dashboard")

if st.button("Start Camera"):
    st.success("Camera button clicked!")

st.info("Dashboard is ready.")

import streamlit as st
import cv2
from ultralytics import YOLO

st.set_page_config(
    page_title="Object Detection Dashboard",
    layout="wide"
)

st.title("🎯 Real-Time Object Detection")
st.write("YOLOv8 + OpenCV + Streamlit")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Start / Stop buttons
start = st.button("📷 Start Camera")

if start:
    stframe = st.empty()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera could not be opened.")
    else:
        st.info("Camera is running. Stop the Streamlit app with Ctrl+C.")

        while True:
            ret, frame = cap.read()

            if not ret:
                st.error("Could not read camera frame.")
                break

            # YOLO object detection
            results = model(frame, verbose=False)

            # Draw boxes and labels
            annotated_frame = results[0].plot()

            # OpenCV uses BGR; Streamlit expects RGB
            annotated_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            # Display frame
            stframe.image(
                annotated_frame,
                channels="RGB",
                use_container_width=True
            )

        cap.release()