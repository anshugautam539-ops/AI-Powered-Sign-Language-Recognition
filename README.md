# AI-Powered-Sign-Language-Recognition
Machine Learning project bridging communication between deaf and hearing individuals
Libraries Used
OpenCV – Real-time video capturing and image processing
MediaPipe – Hand landmark detection (21 key points tracking)
TensorFlow / Keras – Deep Learning model building and training
NumPy – Numerical computations and array handling
Pandas – Dataset preprocessing and data manipulation
Matplotlib – Model accuracy and loss visualization

Model Details
Dataset: Custom hand gesture dataset
Model Type: Convolutional Neural Network
Training Accuracy: ~95% (example value, change if needed)
Real-time prediction performance

Workflow:
Capture live video stream using OpenCV.
Detect and track hand landmarks using MediaPipe (21 key points).
Preprocess landmark coordinates for model compatibility.
Feed processed data into trained CNN classifier.
Display predicted output text on screen in real-time.
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Dummy function for prediction (replace with trained model later)
def predict_gesture(landmarks):
    # Example condition (just demo)
    if landmarks[8][1] < landmarks[6][1]:
        return "Hello"
    else:
        return "Unknown"

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y])

            prediction = predict_gesture(landmarks)

            cv2.putText(frame, prediction, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
