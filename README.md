# Real-Time Object Detection Using YOLOv8

## Project Overview

This project is a real-time object detection application developed using YOLOv8, OpenCV, and Streamlit.

It uses a webcam to detect and track objects in real time. The application displays detected objects with bounding boxes, confidence scores, and tracking IDs.

## Features

- Real-time object detection using webcam
- YOLOv8 object detection model
- Object tracking with tracking IDs
- Confidence threshold adjustment
- Live object count
- Average confidence display
- FPS monitoring
- Detection history and visualization
- Interactive Streamlit dashboard

## Technologies Used

- Python
- YOLOv8
- OpenCV
- Streamlit
- Pandas
- Ultralytics
- Object Tracking

## Model

This project uses the YOLOv8 Nano (`yolov8n.pt`) model for object detection.

The model identifies objects in webcam frames and displays bounding boxes and confidence scores.

## Project Structure

```text
object-detection/
│
├── app.py
├── requirements.txt
├── yolov8n.pt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sumaiyabanu0108/object-detection.git
```

### 2. Navigate to the project folder

```bash
cd object-detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

### 5. Open the application

Open the local URL displayed in your terminal:

`http://localhost:8501`

Allow webcam access when prompted.

## How It Works

1. Start the Streamlit application.
2. Activate the webcam.
3. The YOLOv8 model processes video frames.
4. Objects are detected and displayed with bounding boxes.
5. Tracking IDs and detection statistics are displayed on the dashboard.

## Applications

- Smart surveillance
- Traffic monitoring
- Real-time object tracking
- Computer vision applications
- Automated visual monitoring

## Future Improvements

- Support image and video uploads.
- Improve object tracking performance.
- Add more detection analytics.
- Deploy the application online.

## Author

**Sumaiya Banu**

GitHub: https://github.com/sumaiyabanu0108
