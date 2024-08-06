import warnings
warnings.filterwarnings('ignore', message='SymbolDatabase.GetPrototype() is deprecated')

import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import time

# Resize Images to fit Container
@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(h)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# Function to draw hand landmarks
def draw_hand_landmarks(image, hand_landmarks, mp_hands):
    for landmarks in hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, landmarks, mp_hands.HAND_CONNECTIONS)
    return image

# Initialize MediaPipe Face Mesh and Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Initialize the webcam
video = cv2.VideoCapture(0)

# Create a container for the video feed and warnings
left_column, right_column = st.columns([1, 3])

with right_column:
    st.markdown("### Video Feed")
    video_feed = st.empty()
    st.markdown("### Warnings")
    warning_text = st.empty()

def show_warning(message):
    warning_text.markdown(
        f"<div class='warning'>{message}</div>",
        unsafe_allow_html=True
    )

# Initialize variables for warnings
warnings_count = 0
max_warnings = 3
warning_issued_time = 0
warning_cooldown = 5  # cooldown time in seconds between warnings

with mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5) as face_mesh, mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    prevTime = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(frame_rgb)
        results_hands = hands.process(frame_rgb)

        face_count = 0
        if results_face.multi_face_landmarks:
            face_count = len(results_face.multi_face_landmarks)
            for face_landmarks in results_face.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

        current_time = time.time()
        if face_count == 0 and current_time - warning_issued_time > warning_cooldown:
            warnings_count += 1
            show_warning(f"{warnings_count} - User out of screen!")
            warning_issued_time = current_time

        elif face_count > 2 and current_time - warning_issued_time > warning_cooldown:
            warnings_count += 1
            show_warning(f"More than 1 face detected!")
            warning_issued_time = current_time

        if warnings_count >= max_warnings:
            show_warning("Application terminated due to too many warnings.")
            break

        if results_hands.multi_hand_landmarks:
            frame = draw_hand_landmarks(frame, results_hands.multi_hand_landmarks, mp_hands)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        with right_column:
            video_feed.image(frame, channels='RGB', use_column_width=True)

        time.sleep(0.1)  # Add a short sleep to prevent high CPU usage

video.release()

# Add custom CSS to style the warning message
st.markdown(
    """
    <style>
    .warning {
        background-color: #ffcc00;
        color: #000000;
        padding: 10px;
        border-radius: 5px;
        width: 100%; /* Make this width full */
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)