import cv2
import mediapipe as mp
import time
import numpy as np
import json
from collections import deque

# --- Configuration ---
WINDOW_SIZE = 10

# --- Initialisation MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
RIGHT_IRIS_CENTER = 468

# --- Variables d'état ---
state = "CALIBRATE_NEUTRAL"
calibration_start_time = time.time()
neutral_ratios, down_ratios, up_ratios = [], [], []

thresh_up, thresh_down = 0.0, 0.0
ratio_history = deque(maxlen=WINDOW_SIZE)

cap = cv2.VideoCapture(2)

# Création de la fenêtre et forçage du plein écran
cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def draw_target(img, x, y):
    """Dessine une cible visible à l'écran"""
    cv2.circle(img, (x, y), 15, (0, 0, 255), -1)
    cv2.circle(img, (x, y), 25, (255, 255, 255), 2)
    cv2.circle(img, (x, y), 35, (0, 0, 255), 2)

while cap.isOpened():
    success, image = cap.read()
    if not success: 
        break

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

    if eye_pts:
        cv2.circle(display, eye_pts[0], 3, (0, 255, 255), -1)
        cv2.circle(display, eye_pts[1], 3, (0, 255, 255), -1)
        cv2.circle(display, eye_pts[2], 4, (0, 0, 255), -1)
        cv2.line(display, (eye_pts[2][0], eye_pts[0][1]), (eye_pts[2][0], eye_pts[1][1]), (255, 0, 0), 1)

    if vertical_ratio is not None:
        elapsed = current_time - calibration_start_time

        if state == "CALIBRATE_NEUTRAL":
            draw_target(display, w // 2, h // 2)
            cv2.putText(display, "1. Fixe le CENTRE", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            neutral_ratios.append(vertical_ratio)
            cv2.putText(display, f"{3.0 - elapsed:.1f}s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            if elapsed > 3.0:
                state, calibration_start_time = "CALIBRATE_DOWN", current_time

        elif state == "CALIBRATE_DOWN":
            draw_target(display, w // 2, h - 50)
            cv2.putText(display, "2. Fixe le BAS", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            down_ratios.append(vertical_ratio)
            cv2.putText(display, f"{3.0 - elapsed:.1f}s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            if elapsed > 3.0:
                state, calibration_start_time = "CALIBRATE_UP", current_time

        elif state == "CALIBRATE_UP":
            draw_target(display, w // 2, 50)
            cv2.putText(display, "3. Fixe le HAUT", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            up_ratios.append(vertical_ratio)
            cv2.putText(display, f"{3.0 - elapsed:.1f}s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            if elapsed > 3.0:
                r_neutral = np.mean(neutral_ratios)
                r_down = np.mean(down_ratios)
                r_up = np.mean(up_ratios)
                
                # Calcul des seuils avec une marge stricte
                thresh_down = r_neutral + max((r_down - r_neutral) * 0.75, 0.015)
                thresh_up = r_neutral - max((r_neutral - r_up) * 0.75, 0.015)
                
                state = "RUNNING"
                # Retour au mode fenêtré pour le paramétrage
                cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        elif state == "RUNNING":
            ratio_history.append(vertical_ratio)
            smoothed_ratio = np.mean(ratio_history)

            gaze_zone = "BAS" if smoothed_ratio > thresh_down else ("HAUT" if smoothed_ratio < thresh_up else "MILIEU")

            def draw_ind(text, y, active):
                color, thick, font_s = ((0, 255, 0), 3, 1.2) if active else ((100, 100, 100), 1, 0.8)
                x = (w - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_s, thick)[0][0]) // 2
                cv2.putText(display, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_s, color, thick)

            draw_ind("--- HAUT ---", 40, gaze_zone == "HAUT")
            draw_ind("--- MILIEU ---", h // 2, gaze_zone == "MILIEU")
            draw_ind("--- BAS ---", h - 40, gaze_zone == "BAS")

            cv2.putText(display, f"Ratio : {smoothed_ratio:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, f"Seuil BAS : {thresh_down:.3f} ([a] + / [d] -)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display, f"Seuil HAUT : {thresh_up:.3f} ([z] + / [s] -)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.putText(display, "[v] Valider et Sauvegarder | [r] Recalibrer", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Calibration', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('r') and state == "RUNNING":
        state, neutral_ratios, down_ratios, up_ratios = "CALIBRATE_NEUTRAL", [], [], []
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        calibration_start_time = time.time()
        
    # Touches de réglage (la fenêtre doit être sélectionnée)
    elif key == ord('a') and state == "RUNNING": thresh_down += 0.01  
    elif key == ord('d') and state == "RUNNING": thresh_down -= 0.01  
    elif key == ord('z') and state == "RUNNING": thresh_up += 0.01  
    elif key == ord('s') and state == "RUNNING": thresh_up -= 0.01  
    
    # Touche de validation
    elif key == ord('v') and state == "RUNNING":
        config = {"thresh_down": thresh_down, "thresh_up": thresh_up}
        with open('config.json', 'w') as f:
            json.dump(config, f)
        print("Paramètres sauvegardés dans config.json.")
        break

cap.release()
cv2.destroyAllWindows()