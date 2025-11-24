# Helmet_Detection_In_Traffic
Real-time Helmet Detection system built with YOLOv8, OpenCV, and Supervision.
This project can detect whether a person is wearing a helmet or not in images and videos using a custom-trained YOLOv8 model.

---

## Overview
This project uses a deep learning based object detection model (YOLOv8) to automatically detect helmets in images and video streams.
The goal is to improve road-safety monitoring and workplace compliance by providing fast, accurate, and scalable detection.

---

## Features
- Detects helmets in both images and videos
- Real-time video inference with OpenCV display window
- Automatically saves annotated results (image/video)
- Modular, well-structured Python code (utils + main)
- Built on Ultralytics YOLOv8 + Supervision library for flexible annotation
- Works with custom YOLOv8 models trained on your own dataset

---

## Project Structure
Helmet_Detection_YOLOv8/<br>
│<br>
├── main.py                           # Entry point for running detection<br>
│<br>
├── utils/<br>
│   ├── __init__.py<br>
│   └── helperFunctions.py            # Functions for image and video detection<br>
│<br>
├── models/<br>
│   └── best.pt                       # Custom-trained YOLOv8 model<br>
│<br>
├── input/<br>
│   ├── test_image.jpg<br>
│   └── test_video.mp4<br>
│<br>
├── runs/<br>
│   └── outputs/                      # Annotated results (images/videos)<br>
│<br>
├── requirements.txt<br>
│<br>
└── README.md<br>

---

## Installation
1. Clone this repository<br>
    git clone https://github.com/<your-username>/Helmet_Detection_YOLOv8.git<br>
    cd Helmet_Detection_YOLOv8
2. Create & activate a virtual environment<br>
    python -m venv venv<br>
    venv\Scripts\activate   # On Windows
3. Install dependencies<br>
    pip install -r requirements.txt



