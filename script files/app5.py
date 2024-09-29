import cv2
import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import threading
import streamlit as st
import time

# Constants
SAMPLE_RATE = 44100
DURATION = 5
FRAME_SKIP = 5
VIDEO_CAPTURE_DEVICE = 0

# Model paths
FACIAL_MODEL_PATH = r'D:/emodetector/emodetector/emotiondetector.h5'
VOICE_MODEL_PATH = r'D:/emodetector/emodetector/toronto_speech_model.h5'

# Emotion mapping
EMOTION_MAP = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

class EmotionDetector:
    def __init__(self):
        self.facial_model = self.load_model(FACIAL_MODEL_PATH)
        self.voice_model = self.load_model(VOICE_MODEL_PATH)
        self.predicted_voice_emotion = "Waiting for voice..."
        self.predicted_facial_emotion = "Unknown"

    def load_model(self, path):
        try:
            model = load_model(path)
            return model
        except Exception as e:
            st.error(f"Error loading model from {path}: {e}")
            raise

    def predict_facial_emotion(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (48, 48))
        normalized_frame = resized_frame / 255.0
        input_data = np.expand_dims(np.expand_dims(normalized_frame, -1), 0)
        emotion_prediction = self.facial_model.predict(input_data)
        self.predicted_facial_emotion = EMOTION_MAP[np.argmax(emotion_prediction)]

    def predict_voice_emotion(self, recording):
        audio_features = librosa.feature.mfcc(y=recording.flatten(), sr=SAMPLE_RATE, n_mfcc=13)
        audio_features = np.mean(audio_features.T, axis=0)
        input_data = np.expand_dims(audio_features, axis=0)
        emotion_prediction = self.voice_model.predict(input_data)
        self.predicted_voice_emotion = EMOTION_MAP[np.argmax(emotion_prediction)]

    def display_emotions_on_frame(self, frame):
        cv2.putText(frame, f'Facial Emotion: {self.predicted_facial_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Voice Emotion: {self.predicted_voice_emotion}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def capture_video(self):
        video_capture = cv2.VideoCapture(VIDEO_CAPTURE_DEVICE)
        if not video_capture.isOpened():
            st.error(f"Error opening video capture device: {VIDEO_CAPTURE_DEVICE}")
            return

        frame_count = 0
        placeholder = st.empty()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Error reading from webcam.")
                break

            frame_count += 1

            if frame_count % FRAME_SKIP == 0:
                self.predict_facial_emotion(frame)

            self.display_emotions_on_frame(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB")

            # Add a small delay to control the frame rate
            time.sleep(0.1)

        video_capture.release()

    def record_audio(self):
        while True:
            recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
            sd.wait()  # Wait until recording is finished
            self.predict_voice_emotion(recording)
            st.write(f"Predicted Voice Emotion: {self.predicted_voice_emotion}")

def main():
    st.title("Real-Time Emotion Detection")
    emotion_detector = EmotionDetector()

    # Start audio recording in a separate thread
    audio_thread = threading.Thread(target=emotion_detector.record_audio)
    audio_thread.start()

    # Run video capture in the main thread
    emotion_detector.capture_video()

    # Wait for audio thread to finish
    audio_thread.join()

if __name__ == "__main__":
    main()