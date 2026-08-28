import cv2
import mediapipe as mp
import time
import numpy as np
import json
from collections import deque

# --- Configuration ---
WINDOW_SIZE = 12
CAMERA_INDEX = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=False,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

# Nouveaux repères 2D ultra-stables : Haut du front (10), Bout du nez (1), Bas du menton (152)
FACE_TOP = 10
FACE_NOSE = 1
FACE_CHIN = 152

state = "CALIBRATE_NEUTRAL"
calibration_start_time = time.time()
neutral_ratios, down_ratios = [], []

thresh_down = 0.0
direction = 1
ratio_history = deque(maxlen=WINDOW_SIZE)

cap = cv2.VideoCapture(CAMERA_INDEX)
cv2.namedWindow('Calibration Head', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Calibration Head', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def draw_target(img, x, y):
    cv2.circle(img, (x, y), 15, (0, 0, 255), -1)
    cv2.circle(img, (x, y), 25, (255, 255, 255), 2)
    cv2.circle(img, (x, y), 35, (0, 0, 255), 2)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, c = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    current_time = time.time()
    pitch_ratio = None
    pts = None

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        y_top = landmarks[FACE_TOP].y
        y_nose = landmarks[FACE_NOSE].y
        y_chin = landmarks[FACE_CHIN].y
        
        face_height = y_chin - y_top
        if face_height > 0:
            # Calcul de la position du nez par rapport à la hauteur totale du visage
            pitch_ratio = (y_nose - y_top) / face_height

        pt_top = (int(landmarks[FACE_TOP].x * w), int(landmarks[FACE_TOP].y * h))
        pt_nose = (int(landmarks[FACE_NOSE].x * w), int(landmarks[FACE_NOSE].y * h))
        pt_chin = (int(landmarks[FACE_CHIN].x * w), int(landmarks[FACE_CHIN].y * h))
        pts = (pt_top, pt_nose, pt_chin)

    display = image.copy()
    
    # Affichage des 3 points (Front, Nez, Menton)
    if pts:
        cv2.circle(display, pts[0], 5, (0, 255, 0), -1)
        cv2.circle(display, pts[1], 5, (0, 255, 255), -1)
        cv2.circle(display, pts[2], 5, (0, 255, 0), -1)
        cv2.line(display, pts[0], pts[2], (255, 255, 255), 1)

    if pitch_ratio is not None:
        elapsed = current_time - calibration_start_time

        if state == "CALIBRATE_NEUTRAL":
            draw_target(display, w // 2, h // 2)
            cv2.putText(display, "1. Regarde DROIT DEVANT", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            neutral_ratios.append(pitch_ratio)
            cv2.putText(display, f"{3.0 - elapsed:.1f}s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            if elapsed > 3.0:
                state, calibration_start_time = "CALIBRATE_DOWN", current_time

        elif state == "CALIBRATE_DOWN":
            draw_target(display, w // 2, h - 50)
            cv2.putText(display, "2. Incline la tete vers le BAS", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            down_ratios.append(pitch_ratio)
            cv2.putText(display, f"{3.0 - elapsed:.1f}s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            if elapsed > 3.0:
                r_neutral = np.mean(neutral_ratios)
                r_down = np.mean(down_ratios)
                
                direction = 1 if r_down > r_neutral else -1
                thresh_down = r_neutral + (r_down - r_neutral) * 0.60
                
                state = "RUNNING"
                cv2.setWindowProperty('Calibration Head', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        elif state == "RUNNING":
            ratio_history.append(pitch_ratio)
            smoothed_ratio = np.mean(ratio_history)

            if direction == 1:
                gaze_zone = "ACTION" if smoothed_ratio > thresh_down else "NEUTRE"
            else:
                gaze_zone = "ACTION" if smoothed_ratio < thresh_down else "NEUTRE"

            color = (0, 255, 0) if gaze_zone == "ACTION" else (0, 0, 255)
            font_s = 1.2 if gaze_zone == "ACTION" else 0.8
            text = "--- ACTION SCROLL ---" if gaze_zone == "ACTION" else "--- NEUTRE ---"
            
            x = (w - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_s, 2)[0][0]) // 2
            cv2.putText(display, text, (x, h // 2), cv2.FONT_HERSHEY_SIMPLEX, font_s, color, 2)

            cv2.putText(display, f"Ratio : {smoothed_ratio:.3f} | Seuil : {thresh_down:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, "[a] + Difficile | [d] - Difficile", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display, "[v] Valider et Sauvegarder | [r] Recalibrer | [q] Quitter", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Calibration Head', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('r') and state == "RUNNING":
        state, neutral_ratios, down_ratios = "CALIBRATE_NEUTRAL", [], []
        cv2.setWindowProperty('Calibration Head', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        calibration_start_time = time.time()
        
    elif key == ord('a') and state == "RUNNING": thresh_down += 0.01 * direction  
    elif key == ord('d') and state == "RUNNING": thresh_down -= 0.01 * direction  
    
    elif key == ord('v') and state == "RUNNING":
        config = {"type": "head_pose_ratio", "thresh_down": thresh_down, "direction": direction}
        with open('config.json', 'w') as f:
            json.dump(config, f)
        print("Paramètres sauvegardés avec succès.")
        break

cap.release()
cv2.destroyAllWindows()