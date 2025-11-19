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
Helmet_Detection_YOLOv8/
│
├── main.py                           # Entry point for running detection
│
├── utils/
│   ├── __init__.py
│   └── helperFunctions.py            # Functions for image and video detection
│
├── models/
│   └── best.pt                       # Custom-trained YOLOv8 model
│
├── input/
│   ├── test_image.jpg
│   └── test_video.mp4
│
├── runs/
│   └── outputs/                      # Annotated results (images/videos)
│
├── requirements.txt
│
└── README.md

---

## Installation
1. Clone this repository
    git clone https://github.com/<your-username>/Helmet_Detection_YOLOv8.git<br>
    cd Helmet_Detection_YOLOv8
2. Create & activate a virtual environment
    python -m venv venv<br>
    venv\Scripts\activate   # On Windows
3. Install dependencies
    pip install -r requirements.txt<br>



