# OpenCV Face Recognition Project
![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)
![LBPH](https://img.shields.io/badge/Algorithm-LBPH-lightgrey)

## Features
- **Training Module** (`faces_train.py`):
  - Automated training from image directories
  - Auto-detects person names from folders
  - Haar Cascade face detection
  - Saves model (`face_traind.yml`) and data files

- **Recognition Module** (`face_recognition.py`):
  - Image-based recognition with confidence scores
  - Test image processing with bounding boxes

## Installation
```bash
pip install opencv-contrib-python numpy

## Project Structure
project/
├── Faces/
│   ├── Train/          # Training images
│   └── Test/           # Test images
├── faces_train.py       # Training script
├── face_recognition.py # Recognition script
├── har_face.xml         # Haar Cascade
└── face_traind.yml      # Trained model

## Usage
 1. Training:

    1. Organize training images:
        Faces/Train/
        ├── Person1/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── Person2/
        │   ├── img1.jpg
        │   └── img2.jpg

    2. Run training:
        python faces_train.py
        Outputs:
            face_traind.yml (trained model)
            features.npy and labels.npy

 2. Recognition:
    python face_recognition.py

## Customization
  To Add New People:
    1. Create new folder in Faces/Train/
    2.Add at least 5-10 images per person

  To Improve Accuracy:
    -Increase training images (10-20 per person)
    -Adjust Haar Cascade parameters:
        detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)  # Try higher values