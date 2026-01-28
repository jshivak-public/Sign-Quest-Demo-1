ASL ABC Recognition Pipeline for Sign Quest

This project implement a full end-to-end American Sign Language (ASL) alphabet recognition pipeline developed for the Sign Quest startup. It combines real-time hand tracking, classical machine learning, and webcam-based inference to recognize ASL letters in a lightweight and explainable way.

The system is designed to support Sign Quest’s educational tools by enabling real-time gesture recognition that can later be integrated into games, learning platforms, and interactive demos.

Data Collection

Uses MediaPipe Hands to extract 21 hand landmarks per frame.

Each frame is converted into a 63-dimensional feature vector (x, y, z for each landmark).

Supports both:

Webcam-based data collection with optional key labeling

Pre-recorded video input with automatic class labeling

All collected samples are stored in a CSV file (asl_abc_data.csv) to keep the dataset transparent and easy to audit.

Model Training

Trains a Logistic Regression classifier using scikit-learn.

Training pipeline includes:

Feature normalization with StandardScaler

Class-balanced logistic regression for improved stability

Automatically splits data into training and test sets and prints a classification report.

The trained model, class list, and confidence threshold are saved together in a serialized bundle (model.pk1) for consistent inference.

Real-Time Inference

Runs live inference from a webcam feed.

Displays:

Detected hand landmarks

Predicted ASL character

Confidence score for the prediction

Includes an unknown-class rejection mechanism:

If the highest prediction probability falls below a configurable threshold, the output is labeled as unknown.

The threshold can be adjusted at runtime using keyboard input.

Shows raw and annotated frames side-by-side to help with debugging and demos.

Key Design Goals

Real-time performance on consumer hardware

Explainable, non–black-box ML for demos and education

Simple data pipeline that supports iteration and expansion

Easy integration into Sign Quest prototypes and presentations

Tech Stack

Python

OpenCV

MediaPipe

NumPy and Pandas

Scikit-learn

cvzone

Notes

Its designed for single-hand ASL alphabet recognition.

The pipeline can be extended to additional signs, gestures, or multi-hand inputs as Sign Quest evolves.

This project serves as a strong baseline system rather than a deep learning solution, prioritizing clarity and control over complexity.
