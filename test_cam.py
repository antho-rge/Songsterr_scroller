import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque

# --- Configuration ---
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
RIGHT_IRIS_CENTER = 468

# --- Variables d'état ---
state = "CALIBRATE_NEUTRAL"
calibration_start_time = time.time()
neutral_ratios = []
down_ratios = []

thresh_up = 0.0
thresh_down = 0.0

ratio_history = deque(maxlen=WINDOW_SIZE)

cap = cv2.VideoCapture(0)

# Mettre la fenêtre en mode redimensionnable pour que tu puisses l'agrandir
cv2.namedWindow('Eye Tracking Debug', cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Miroir horizontal
    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    current_time = time.time()
    vertical_ratio = None
    eye_pts = None

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        y_top = landmarks[RIGHT_EYE_TOP].y
        y_bottom = landmarks[RIGHT_EYE_BOTTOM].y
        y_iris = landmarks[RIGHT_IRIS_CENTER].y

        eye_height = y_bottom - y_top
        if eye_height > 0:
            vertical_ratio = (y_iris - y_top) / eye_height

        pt_top = (int(landmarks[RIGHT_EYE_TOP].x * w), int(landmarks[RIGHT_EYE_TOP].y * h))
        pt_bottom = (int(landmarks[RIGHT_EYE_BOTTOM].x * w), int(landmarks[RIGHT_EYE_BOTTOM].y * h))
        pt_iris = (int(landmarks[RIGHT_IRIS_CENTER].x * w), int(landmarks[RIGHT_IRIS_CENTER].y * h))
        eye_pts = (pt_top, pt_bottom, pt_iris)

    display = image.copy()

    # Dessin des repères
    if eye_pts:
        p_top, p_bottom, p_iris = eye_pts
        cv2.circle(display, p_top, 3, (0, 255, 255), -1)
        cv2.circle(display, p_bottom, 3, (0, 255, 255), -1)
        cv2.circle(display, p_iris, 4, (0, 0, 255), -1)
        cv2.line(display, (p_iris[0], p_top[1]), (p_iris[0], p_bottom[1]), (255, 0, 0), 1)

    if vertical_ratio is not None:
        if state == "CALIBRATE_NEUTRAL":
            cv2.putText(display, "1. Fixe le CENTRE de l'ecran", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            neutral_ratios.append(vertical_ratio)
            
            elapsed = current_time - calibration_start_time
            cv2.putText(display, f"Temps : {3.0 - elapsed:.1f}s", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if elapsed > 3.0:
                state = "CALIBRATE_DOWN"
                calibration_start_time = current_time

        elif state == "CALIBRATE_DOWN":
            cv2.putText(display, "2. Fixe le BAS de l'ecran", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            down_ratios.append(vertical_ratio)
            
            elapsed = current_time - calibration_start_time
            cv2.putText(display, f"Temps : {3.0 - elapsed:.1f}s", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if elapsed > 3.0:
                r_neutral = np.mean(neutral_ratios)
                r_down = np.mean(down_ratios)
                
                # Calcul des seuils avec une marge stricte pour éviter les déclenchements parasites
                marge = max((r_down - r_neutral) * 0.75, 0.015)
                thresh_down = r_neutral + marge
                thresh_up = r_neutral - marge # On déduit le haut symétriquement
                
                state = "RUNNING"

        elif state == "RUNNING":
            ratio_history.append(vertical_ratio)
            smoothed_ratio = np.mean(ratio_history)

            # Détermination de la zone regardée
            if smoothed_ratio > thresh_down:
                gaze_zone = "BAS"
            elif smoothed_ratio < thresh_up:
                gaze_zone = "HAUT"
            else:
                gaze_zone = "MILIEU"

            # --- Affichage des indicateurs HAUT / MILIEU / BAS ---
            def draw_indicator(text, y_pos, is_active):
                color = (0, 255, 0) if is_active else (100, 100, 100)
                thickness = 3 if is_active else 1
                font_scale = 1.2 if is_active else 0.8
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                x_pos = (w - text_size[0]) // 2 # Centrage horizontal
                cv2.putText(display, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            draw_indicator("--- HAUT ---", 40, gaze_zone == "HAUT")
            draw_indicator("--- MILIEU ---", h // 2, gaze_zone == "MILIEU")
            draw_indicator("--- BAS ---", h - 20, gaze_zone == "BAS")

            # Infos de debug en haut à gauche
            cv2.putText(display, f"Ratio : {smoothed_ratio:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, "r: Recalibrer | q: Quitter", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

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