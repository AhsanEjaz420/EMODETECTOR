# EMODETECTOR
Emotion Detector
This repository contains a multi-modal emotion detection system that analyzes facial expressions and speech data to recognize emotions in real time. The system leverages deep learning models, including Convolutional Neural Networks (CNN) for facial analysis and Recurrent Neural Networks (RNN) for speech analysis, providing a robust solution for detecting emotions across multiple modalities.

Table of Contents
Introduction
Features
Installation
Usage
Dataset
Model Architecture
Evaluation
Results
Contributing
License
Introduction
Emotion detection plays a crucial role in various applications such as sentiment analysis, user experience enhancement, and human-computer interaction. This project combines facial expression recognition and speech analysis to classify emotions such as happiness, sadness, anger, surprise, and others.

The model analyzes input from both images (facial expressions) and audio (speech signals) to predict the corresponding emotion, making it more accurate than single-modal systems.

Features
Multi-modal Emotion Detection: Combines facial and speech data for enhanced accuracy.
Deep Learning Models: Uses CNN for image-based facial expression analysis and RNN for speech analysis.
Real-time Analysis: Capable of real-time emotion detection from live video and audio streams.
Performance Metrics: Evaluated using precision, recall, and F1-score to ensure robust performance.
Cross-emotion Accuracy: Detects multiple emotions with high accuracy.
Model Architecture
Facial Expression Model (CNN)
Input: Grayscale image (48x48 pixels)
Architecture: 5 convolutional layers with ReLU activations followed by max pooling and dropout layers.
Output: Softmax layer with 7 classes representing different emotions.
Speech Emotion Model (RNN)
Input: MFCC features extracted from speech audio.
Architecture: 2 LSTM layers followed by a dense layer with softmax output.
Output: Softmax layer with emotion predictions for speech.
Combined Model
Fusion: The outputs from the CNN and RNN are concatenated and passed through a fully connected layer to produce the final prediction.
Evaluation
The model is evaluated using the following metrics:

Precision: Measures the accuracy of the positive predictions.
Recall: Measures the completeness of the positive predictions.
F1-score: A balanced metric combining precision and recall.
Results
The multi-modal emotion detection system achieved the following results:

Facial Expression Model Accuracy: 95%
Speech Model Accuracy: 90%
Combined Model Accuracy: 97%
F1-score: 0.96
These results demonstrate the effectiveness of combining facial and speech data for emotion detection.

Contributing
Contributions are welcome! If you’d like to contribute, please fork the repository and submit a pull request with detailed explanations of your changes.

License
This project is licensed under the MIT License. See the LICENSE file for more information.

