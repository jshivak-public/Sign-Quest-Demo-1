# ASL ABC Recognition Pipeline for Sign Quest

This project implement a full end-to-end American Sign Language (ASL) alphabet recognition pipeline developed for Sign Quest, an ASL ed-tech startup focussed on providing DHH families and educators with accesible resources to teach students American Sign Language.

The system is designed to support Sign Quest’s educational tools by enabling real-time gesture recognition, interactive demos, and rapid prototyping of ASL-based learning experiences.

## Overview

The pipeline follows a simple and transparent workflow:

- Capture hand pose data using a webcam or pre-recorded video
- Extract 3D hand landmark features using MediaPipe
- Train a lightweight machine learning classifier on labeled ASL data
- Run real-time inference with confidence-based unknown rejection

The focus of this project is clarity and reliability rather than black-box deep learning models, making it suitable for education-focused applications and live demonstrations.

## Data Collection

- Uses MediaPipe Hands to extract 21 hand landmarks per frame.
- Each frame is converted into a 63-dimensional feature vector (x, y, z for each landmark).
- Only a single hand is tracked at a time to reduce ambiguity.
- Supports both:
  - Webcam-based data collection with optional key labeling
  - Pre-recorded video input with automatic class labeling
- Samples are appended row-by-row to a CSV dataset.
- Data is stored in `asl_abc_data.csv` for easy inspection and reuse.

## Model Training

- Trains a Logistic Regression classifier using scikit-learn.
- A preprocessing and model pipeline is used to ensure consistent inference.
- Training includes:
  - Feature normalization using `StandardScaler`
  - Class-balanced logistic regression to handle uneven datasets
- The dataset is split into training and test sets for evaluation.
- A classification report is printed after training to measure performance.
- The trained model, class labels, and configuration settings are saved to `model.pk1`.

## Real-Time Inference

- Performs live ASL letter recognition using a webcam feed.
- Displays:
  - Detected hand landmarks
  - Predicted ASL letter
  - Model confidence score
- Implements confidence-based unknown detection:
  - If the highest prediction probability falls below a threshold, the output is labeled as `unknown`.
  - This helps reject unclear or partially formed signs.
- The confidence threshold can be adjusted at runtime using keyboard input.
- Raw and annotated frames are displayed side-by-side to support debugging and demos.

## Design Decisions

- Uses classical machine learning instead of deep learning to remain interpretable.
- Relies on MediaPipe landmarks rather than raw image pixels.
- Keeps the data pipeline CSV-based to simplify iteration and debugging.
- Designed for real-time performance on consumer hardware.

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- cvzone

## Notes

- This system is currently designed for single-hand ASL alphabet recognition.
- Additional signs, gestures, or multi-hand support can be added as Sign Quest expands.
- The project serves as a baseline recognition system that can be integrated into future Sign Quest products and demos.
