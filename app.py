import streamlit as st
import cv2
import time
import pandas as pd
from ultralytics import YOLO

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🎯",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

# -----------------------------
# Session State
# -----------------------------
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "last_detections" not in st.session_state:
    st.session_state.last_detections = []

if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

# -----------------------------
# Header
# -----------------------------
st.title("🎯 AI Real-Time Object Detection")
st.caption("YOLOv8 + OpenCV + Streamlit | Local Web Dashboard")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Detection Settings")

dark_mode = st.sidebar.toggle("Dark Mode", value=True)

if dark_mode:
    st.markdown(
        """
        <style>
        .stApp { background: #101418; color: #f1f5f9; }
        [data-testid="stSidebar"] { background: #171d23; }
        [data-testid="stSidebar"] * { color: #f1f5f9; }
        [data-testid="stMetric"] {
            background: #1b242c;
            border: 1px solid #33404b;
            padding: 12px;
            border-radius: 8px;
        }
        [data-testid="stDataFrame"] { border: 1px solid #33404b; }
        .stInfo { background: #172b3b; color: #dbeafe; }
        </style>
        """,
        unsafe_allow_html=True,
    )

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.10,
    1.00,
    0.50,
    0.05
)

tracking_enabled = st.sidebar.checkbox("Enable Object Tracking", value=True)

if st.sidebar.button("Clear Detection History", use_container_width=True):
    st.session_state.detection_history = []

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Model Performance")
st.sidebar.metric("mAP@50", "52.6%")
st.sidebar.metric("mAP@50-95", "37.4%")
st.sidebar.metric("Recall@100", "58.9%")

# -----------------------------
# Start / Stop
# -----------------------------
col_start, col_stop = st.columns(2)

with col_start:
    if st.button("▶️ Start Camera", use_container_width=True):
        st.session_state.camera_running = True

with col_stop:
    if st.button("⏹️ Stop Camera", use_container_width=True):
        st.session_state.camera_running = False

# -----------------------------
# Camera
# -----------------------------
if st.session_state.camera_running:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Camera could not be opened.")
        st.session_state.camera_running = False
        st.stop()

    frame_placeholder = st.empty()

    metric1, metric2, metric3 = st.columns(3)

    object_metric = metric1.empty()
    confidence_metric = metric2.empty()
    fps_metric = metric3.empty()

    table_placeholder = st.empty()
    history_placeholder = st.empty()

    previous_time = time.time()

    while st.session_state.camera_running:

        success, frame = cap.read()

        if not success:
            st.error("❌ Could not read camera frame.")
            break

        # YOLO detection
        if tracking_enabled:
            results = model.track(
                frame,
                conf=confidence_threshold,
                persist=True,
                verbose=False
            )
        else:
            results = model(
                frame,
                conf=confidence_threshold,
                verbose=False
            )

        result = results[0]

        # Draw bounding boxes
        annotated_frame = result.plot()

        # Detection information
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0]) * 100

            class_name = model.names[class_id]

            track_id = "-"
            if tracking_enabled and box.id is not None:
                track_id = str(int(box.id[0]))

            detections.append({
                "Object": class_name,
                "Track ID": track_id,
                "Confidence": f"{confidence:.1f}%"
            })

        # Object count
        object_count = len(detections)

        # Average confidence
        if object_count > 0:
            average_confidence = sum(
                float(item["Confidence"].replace("%", ""))
                for item in detections
            ) / object_count
        else:
            average_confidence = 0

        st.session_state.detection_history.append({
            "Frame": len(st.session_state.detection_history) + 1,
            "Objects": object_count,
            "Average Confidence": round(average_confidence, 1),
        })
        st.session_state.detection_history = st.session_state.detection_history[-20:]

        # FPS
        current_time = time.time()
        fps = 1 / max(current_time - previous_time, 0.001)
        previous_time = current_time

        # Metrics
        object_metric.metric(
            "🔢 Objects Detected",
            object_count
        )

        confidence_metric.metric(
            "🎯 Avg Confidence",
            f"{average_confidence:.1f}%"
        )

        fps_metric.metric(
            "⚡ FPS",
            f"{fps:.1f}"
        )

        # Convert BGR → RGB
        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        # Display camera
        frame_placeholder.image(
            annotated_frame,
            channels="RGB",
            use_container_width=True
        )

        # Display detected objects
        if detections:
            table_placeholder.dataframe(
                pd.DataFrame(detections),
                use_container_width=True,
                hide_index=True
            )
        else:
            table_placeholder.info("No objects detected.")

        history_placeholder.line_chart(
            pd.DataFrame(st.session_state.detection_history).set_index("Frame")[[
                "Objects",
                "Average Confidence",
            ]],
            use_container_width=True,
        )

    cap.release()

else:

    st.info("👆 Click **Start Camera** to begin detection.")

    st.markdown("---")

    # Model performance cards
    col1, col2, col3 = st.columns(3)

    col1.metric("mAP@50", "52.6%")
    col2.metric("mAP@50-95", "37.4%")
    col3.metric("Recall@100", "58.9%")

    st.markdown("---")

    st.subheader("📌 Detection Pipeline")

    st.write(
        """
        **📷 Webcam → OpenCV → 🤖 YOLOv8 → 🎯 Detection → 🌐 Local Dashboard**

        All processing is performed locally on this computer.
        """
    )

    st.markdown("---")
    st.subheader("📈 Detection History")

    if st.session_state.detection_history:
        history_df = pd.DataFrame(st.session_state.detection_history)
        history_metric1, history_metric2, history_metric3 = st.columns(3)

        history_metric1.metric("Frames Recorded", len(history_df))
        history_metric2.metric("Peak Objects", int(history_df["Objects"].max()))
        history_metric3.metric(
            "Average Confidence",
            f"{history_df['Average Confidence'].mean():.1f}%",
        )

        st.line_chart(
            history_df.set_index("Frame")[["Objects", "Average Confidence"]],
            use_container_width=True,
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No detection history yet. Start the camera to record detections.")