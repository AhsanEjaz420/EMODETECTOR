import cv2
import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import threading
import matplotlib.pyplot as plt

# Constants
SAMPLE_RATE = 44100
DURATION = 5
FRAME_SKIP = 5
VIDEO_CAPTURE_DEVICE = 0

# Load models for emotion detection
facial_model = load_model(r'D:/emodetector/emodetector/emotiondetector.h5')
voice_model = load_model(r'D:/emodetector/emodetector/toronto_speech_model.h5')

# Mapping for emotions
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Global variable to hold the predicted voice emotion
predicted_voice_emotion = "Waiting for voice..."

# Initialize predicted_facial_emotion to avoid reference issues
predicted_facial_emotion = "Unknown"  # Default until first prediction is made

def capture_video():
    global predicted_voice_emotion
    global predicted_facial_emotion

    video_capture = cv2.VideoCapture(VIDEO_CAPTURE_DEVICE)
    
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error reading from webcam.")
            break

        frame_count += 1

        # Process every nth frame for facial emotion detection
        if frame_count % FRAME_SKIP == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (48, 48))
            normalized_frame = resized_frame / 255.0
            input_data = np.expand_dims(np.expand_dims(normalized_frame, -1), 0)

            # Predict facial emotion
            emotion_prediction = facial_model.predict(input_data)
            predicted_facial_emotion = emotion_map[np.argmax(emotion_prediction)]

        # Display the predicted emotions on the frame
        display_emotions_on_frame(frame)

        # Convert frame from BGR to RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Clear the previous image and display the new frame
        ax.clear()
        ax.imshow(frame_rgb)
        ax.axis('off')  # Hide axes
        plt.draw()

        # Use plt.pause to allow the UI to refresh
        plt.pause(0.001)

    video_capture.release()
    plt.close(fig)

def display_emotions_on_frame(frame):
    cv2.putText(frame, f'Facial Emotion: {predicted_facial_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Voice Emotion: {predicted_voice_emotion}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def record_audio():
    global predicted_voice_emotion

    while True:
        print("Recording Audio...")
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()  # Wait until recording is finished
        
        # Extract MFCC features from the audio
        audio_features = librosa.feature.mfcc (y=recording.flatten(), sr=SAMPLE_RATE, n_mfcc=13)
        audio_features = np.mean(audio_features.T, axis=0)
        input_data = np.expand_dims(audio_features, axis=0)

        # Predict voice emotion
        emotion_prediction = voice_model.predict(input_data)
        predicted_voice_emotion = emotion_map[np.argmax(emotion_prediction)]
        print(f"Predicted Voice Emotion: {predicted_voice_emotion}")

# Start audio recording in a separate thread
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

# Run video capture in the main thread
capture_video()

# Wait for audio thread to finish
audio_thread.join()