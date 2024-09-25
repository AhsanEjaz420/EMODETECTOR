Emotion Detector

Table of Contents

- #overview
- #requirements
- #models
- #usage
- #running-the-application

Overview

This repository contains an Emotion Detector application built using Streamlit. The application utilizes two machine learning models to detect emotions from facial expressions and speech.

Requirements

- Python 3.8+
- Streamlit 1.12+
- TensorFlow 2.8+
- OpenCV 4.5+
- Librosa 0.9+

Models

Facial Expression Model

- Model Name: Emotion Detector
- Model Path: emotiondetector.h5
- Description: A convolutional neural network (CNN) model trained on the FER2013 dataset to detect emotions from facial expressions.

Speech Model

- Model Name: Toronto Speech Model
- Model Path: toronto_speech_model.h5
- Description: A recurrent neural network (LSTM) model trained on the Toronto Emotional Speech Dataset to detect emotions from speech.

Usage

Running the Application


To run the application:


1. Navigate to the script directory in Command Prompt (Cmd).
2. Replace the model paths with your actual paths:


FACIAL_MODEL_PATH = r'D:/emodetector/emodetector/emotiondetector.h5'
VOICE_MODEL_PATH = r'D:/emodetector/emodetector/toronto_speech_model.h5'


1. Run the following command:
 
streamlit run your_script_name.py

The application will launch in your default web browser.

Note: Ensure you have the necessary dependencies installed and the model paths are correct to run the application successfully.