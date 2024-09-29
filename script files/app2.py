import cv2
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import streamlit as st
from tensorflow.keras.models import load_model
import threading

# Load models for emotion detection
facial_model = load_model(r'D:/emodetector/emodetector/emotiondetector.h5')
voice_model = load_model(r'D:/emodetector/emodetector/toronto_speech_model.h5')


# Mapping for emotions
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False

# Create placeholders for displaying video and emotions
video_placeholder = st.empty()
facial_emotion_placeholder = st.empty()
voice_emotion_placeholder = st.empty()

# Function to capture video and predict facial emotion
def capture_video():
    video_capture = cv2.VideoCapture(0)
    while st.session_state.running:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Error reading from webcam.")
            break

        # Preprocess the frame for facial emotion detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (48, 48))
        normalized_frame = resized_frame / 255.0
        input_data = np.expand_dims(np.expand_dims(normalized_frame, -1), 0)

        # Predict facial emotion
        emotion_prediction = facial_model.predict(input_data)
        predicted_facial_emotion = emotion_map[np.argmax(emotion_prediction)]

        # Display the predicted facial emotion on the frame
        frame = cv2.putText(frame, f'Facial Emotion: {predicted_facial_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame and facial emotion in Streamlit
        video_placeholder.image(frame_rgb, channels="RGB")
        facial_emotion_placeholder.text(f"Predicted Facial Emotion: {predicted_facial_emotion}")

    video_capture.release()

# Function to record audio and predict voice emotion
def record_audio():
    fs = 44100  # Sample rate
    duration = 5  # Duration of recording

    while st.session_state.running:
        # Record audio
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        # Save the audio to a temporary file
        wav_file = 'temp_audio.wav'
        sf.write(wav_file, recording, fs)

        # Extract features from the audio
        audio_features = librosa.feature.mfcc(y=recording.flatten(), sr=fs, n_mfcc=13)
        audio_features = np.mean(audio_features.T, axis=0)
        input_data = np.expand_dims(audio_features, axis=0)

        # Predict voice emotion
        emotion_prediction = voice_model.predict(input_data)
        predicted_voice_emotion = emotion_map[np.argmax(emotion_prediction)]

        # Display the predicted voice emotion and play the audio in Streamlit
        voice_emotion_placeholder.text(f"Predicted Voice Emotion: {predicted_voice_emotion}")
        st.audio(wav_file, format="audio/wav")

# Streamlit interface
st.title("Real-time Emotion Detection")
st.subheader("Facial and Voice Emotion Detection")

if st.button('Start Detection'):
    st.session_state.running = True
    # Start video and audio capture in separate threads
    threading.Thread(target=capture_video).start()
    threading.Thread(target=record_audio).start()

if st.button('Stop Detection'):
    st.session_state.running = False
