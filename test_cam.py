import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque

# --- Configuration ---
SCROLL_KEY = 'pagedown'
FRAMES_TO_TRIGGER = 8
COOLDOWN_TIME = 2.0
WINDOW_SIZE = 10

# --- Initialisation MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Landmarks œil droit
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
RIGHT_IRIS_CENTER = 473

# --- Variables d'état ---
state = "CALIBRATE_NEUTRAL"
calibration_start_time = time.time()
neutral_ratios = []
down_ratios = []
gaze_threshold = 0.0

ratio_history = deque(maxlen=WINDOW_SIZE)
consecutive_frames = 0
last_scroll_time = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Miroir horizontal pour un retour caméra plus intuitif
    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    current_time = time.time()
    vertical_ratio = None
    eye_pts = None

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Coordonnées normalisées
        y_top = landmarks[RIGHT_EYE_TOP].y
        y_bottom = landmarks[RIGHT_EYE_BOTTOM].y
        y_iris = landmarks[RIGHT_IRIS_CENTER].y

        eye_height = y_bottom - y_top
        if eye_height > 0:
            vertical_ratio = (y_iris - y_top) / eye_height

        # Coordonnées en pixels pour le rendu visuel
        pt_top = (int(landmarks[RIGHT_EYE_TOP].x * w), int(landmarks[RIGHT_EYE_TOP].y * h))
        pt_bottom = (int(landmarks[RIGHT_EYE_BOTTOM].x * w), int(landmarks[RIGHT_EYE_BOTTOM].y * h))
        pt_iris = (int(landmarks[RIGHT_IRIS_CENTER].x * w), int(landmarks[RIGHT_IRIS_CENTER].y * h))
        eye_pts = (pt_top, pt_bottom, pt_iris)

    display = image.copy()

    # Dessin des repères sur le flux vidéo
    if eye_pts:
        p_top, p_bottom, p_iris = eye_pts
        cv2.circle(display, p_top, 3, (0, 255, 255), -1)
        cv2.circle(display, p_bottom, 3, (0, 255, 255), -1)
        cv2.circle(display, p_iris, 4, (0, 0, 255), -1)  # Iris en rouge
        
        # Vecteur directionnel vertical
        cv2.line(display, (p_iris[0], p_top[1]), (p_iris[0], p_bottom[1]), (255, 0, 0), 1)

    # Machine à états
    if vertical_ratio is not None:
        if state == "CALIBRATE_NEUTRAL":
            cv2.putText(display, "1. Regarde normalement (Neutre)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            neutral_ratios.append(vertical_ratio)
            
            elapsed = current_time - calibration_start_time
            cv2.putText(display, f"Temps : {3.0 - elapsed:.1f}s", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if elapsed > 3.0:
                state = "CALIBRATE_DOWN"
                calibration_start_time = current_time

        elif state == "CALIBRATE_DOWN":
            cv2.putText(display, "2. Regarde le BAS de l'ecran", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            down_ratios.append(vertical_ratio)
            
            elapsed = current_time - calibration_start_time
            cv2.putText(display, f"Temps : {3.0 - elapsed:.1f}s", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if elapsed > 3.0:
                r_neutral = np.mean(neutral_ratios)
                r_down = np.mean(down_ratios)
                gaze_threshold = r_neutral + (r_down - r_neutral) * 0.6
                state = "RUNNING"

        elif state == "RUNNING":
            ratio_history.append(vertical_ratio)
            smoothed_ratio = np.mean(ratio_history)

            # Indicateur visuel d'état
            is_looking_down = smoothed_ratio > gaze_threshold
            color = (0, 0, 255) if is_looking_down else (0, 255, 0)
            
            cv2.putText(display, f"Ratio: {smoothed_ratio:.2f} (Seuil: {gaze_threshold:.2f})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Barre de progression visuelle pour le trigger
            bar_width = int((consecutive_frames / FRAMES_TO_TRIGGER) * 200)
            cv2.rectangle(display, (30, 60), (230, 75), (100, 100, 100), 1)
            if bar_width > 0:
                cv2.rectangle(display, (30, 60), (30 + bar_width, 75), (0, 0, 255), -1)

            if current_time - last_scroll_time > COOLDOWN_TIME:
                if is_looking_down:
                    consecutive_frames += 1
                else:
                    consecutive_frames = 0

                if consecutive_frames >= FRAMES_TO_TRIGGER:
                    pyautogui.press(SCROLL_KEY)
                    last_scroll_time = current_time
                    consecutive_frames = 0
            else:
                cv2.putText(display, "COOLDOWN", (240, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            cv2.putText(display, "r: Recalibrer | q: Quitter", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow('Eye Tracking Debug', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and state == "RUNNING":
        state = "CALIBRATE_NEUTRAL"
        neutral_ratios.clear()
        down_ratios.clear()
        calibration_start_time = time.time()

cap.release()
cv2.destroyAllWindows()