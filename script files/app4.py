import cv2
import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import threading
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            logging.info(f"Model loaded successfully from {path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model from {path}: {e}")
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
            logging.error(f"Error opening video capture device: {VIDEO_CAPTURE_DEVICE}")
            return

        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()

        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                logging.error("Error reading from webcam.")
                break

            frame_count += 1

            if frame_count % FRAME_SKIP == 0:
                self.predict_facial_emotion(frame)

            self.display_emotions_on_frame(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.clear()
            ax.imshow(frame_rgb)
            ax.axis('off')  # Hide axes
            plt.draw()
            plt.pause(0.001)

        video_capture.release()
        plt.close(fig)

    def record_audio(self):
        while True:
            print("Recording Audio...")
            recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
            sd.wait()  # Wait until recording is finished
            self.predict_voice_emotion(recording)
            print(f"Predicted Voice Emotion: {self.predicted_voice_emotion}")

def main():
    emotion_detector = EmotionDetector()
    audio_thread = threading.Thread(target=emotion_detector.record_audio)
    audio_thread.start()
    emotion_detector.capture_video()
    audio_thread.join()

if __name__ == "__main__":
    main()