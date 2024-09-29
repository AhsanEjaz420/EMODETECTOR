import cv2
import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import threading
import matplotlib.pyplot as plt

# Load models for emotion detection
facial_model = load_model(r'D:/emodetector/emodetector/emotiondetector.h5')
voice_model = load_model(r'D:/emodetector/emodetector/toronto_speech_model.h5')


# Mapping for emotions
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Global variable to hold the predicted voice emotion
predicted_voice_emotion = "Waiting for voice..."

# Frame skipping value (adjust as necessary)
frame_skip = 5  # Predict every 5th frame
frame_count = 0  # Counter to track the number of frames processed

# Initialize predicted_facial_emotion to avoid reference issues
predicted_facial_emotion = "Unknown"  # Default until first prediction is made

# Function to capture video and predict facial emotions
def capture_video():
    global predicted_voice_emotion
    global frame_count
    global predicted_facial_emotion  # Add this to access the global variable

    video_capture = cv2.VideoCapture(0)
    
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error reading from webcam.")
            break

        frame_count += 1  # Increment frame counter

        # Process every nth frame for facial emotion detection
        if frame_count % frame_skip == 0:  # Process every nth frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (48, 48))
            normalized_frame = resized_frame / 255.0
            input_data = np.expand_dims(np.expand_dims(normalized_frame, -1), 0)

            # Predict facial emotion
            emotion_prediction = facial_model.predict(input_data)
            predicted_facial_emotion = emotion_map[np.argmax(emotion_prediction)]  # Update the global variable

        # Display the predicted facial emotion on the frame (uses the last predicted value when skipping frames)
        cv2.putText(frame, f'Facial Emotion: {predicted_facial_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the predicted voice emotion on the frame
        cv2.putText(frame, f'Voice Emotion: {predicted_voice_emotion}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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

# Function to record audio and predict voice emotions
def record_audio():
    global predicted_voice_emotion
    fs = 44100  # Sample rate
    duration = 5  # Record for 5 seconds
    
    while True:
        print("Recording Audio...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        
        # Extract MFCC features from the audio
        audio_features = librosa.feature.mfcc(y=recording.flatten(), sr=fs, n_mfcc=13)
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
