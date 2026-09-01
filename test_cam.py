import cv2
import mediapipe as mp
import time
import numpy as np
import json
import winsound
from collections import deque

WINDOW_SIZE = 3
CAMERA_INDEX = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

FACE_TOP = 10
FACE_NOSE = 1
FACE_CHIN = 152
EYE_LEFT_OUTER = 263
EYE_RIGHT_OUTER = 33

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    dist_v1 = np.linalg.norm(pts[1] - pts[5])
    dist_v2 = np.linalg.norm(pts[2] - pts[4])
    dist_h = np.linalg.norm(pts[0] - pts[3])
    return (dist_v1 + dist_v2) / (2.0 * dist_h) if dist_h > 0 else 0.0

state = "CALIBRATE_NEUTRAL"
calibration_start_time = time.time()
neutral_ratios, down_ratios, up_ratios = [], [], []
neutral_yaw_ratios = []

thresh_down, thresh_up = 0.0, 0.0
r_neutral = 0.0
direction = 1
yaw_neutral = 0.50
yaw_tolerance = 0.25

ratio_history = deque(maxlen=WINDOW_SIZE)

is_paused = False
eyes_closed_start = None

cap = cv2.VideoCapture(CAMERA_INDEX)
cv2.namedWindow('Calibration Head & Gaze', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Calibration Head & Gaze', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def draw_target(img, x, y):
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
    pitch_ratio = None
    yaw_ratio = None
    eyes_closed = False

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        y_top = landmarks[FACE_TOP].y
        y_nose = landmarks[FACE_NOSE].y
        y_chin = landmarks[FACE_CHIN].y
        
        face_height = y_chin - y_top
        if face_height > 0:
            pitch_ratio = (y_nose - y_top) / face_height

        x_left = landmarks[EYE_LEFT_OUTER].x
        x_right = landmarks[EYE_RIGHT_OUTER].x
        x_nose = landmarks[FACE_NOSE].x
        eye_dist = abs(x_left - x_right)
        if eye_dist > 0:
            yaw_ratio = (x_nose - min(x_left, x_right)) / eye_dist

        ear_left = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (ear_left + ear_right) / 2.0
        eyes_closed = avg_ear < 0.18

    display = image.copy()

    if pitch_ratio is not None and yaw_ratio is not None:
        elapsed = current_time - calibration_start_time

        if state == "CALIBRATE_NEUTRAL":
            draw_target(display, w // 2, h // 2)
            cv2.putText(display, "1. Regarde DROIT DEVANT (Centre)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            neutral_ratios.append(pitch_ratio)
            neutral_yaw_ratios.append(yaw_ratio)
            cv2.putText(display, f"{3.0 - elapsed:.1f}s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            if elapsed > 3.0:
                state, calibration_start_time = "CALIBRATE_DOWN", current_time

        elif state == "CALIBRATE_DOWN":
            draw_target(display, w // 2, h - 50)
            cv2.putText(display, "2. Incline la tete vers le BAS", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            down_ratios.append(pitch_ratio)
            cv2.putText(display, f"{3.0 - elapsed:.1f}s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            if elapsed > 3.0:
                state, calibration_start_time = "CALIBRATE_UP", current_time

        elif state == "CALIBRATE_UP":
            draw_target(display, w // 2, 50)
            cv2.putText(display, "3. Leve la tete vers le HAUT", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            up_ratios.append(pitch_ratio)
            cv2.putText(display, f"{3.0 - elapsed:.1f}s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            if elapsed > 3.0:
                r_neutral = float(np.mean(neutral_ratios))
                r_down = float(np.mean(down_ratios))
                r_up = float(np.mean(up_ratios))
                yaw_neutral = float(np.mean(neutral_yaw_ratios))
                
                direction = 1 if r_down > r_neutral else -1
                thresh_down = r_neutral + (r_down - r_neutral) * 0.60
                thresh_up = r_neutral + (r_up - r_neutral) * 0.60
                
                state = "RUNNING"
                cv2.setWindowProperty('Calibration Head & Gaze', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        elif state == "RUNNING":
            if eyes_closed:
                if eyes_closed_start is None:
                    eyes_closed_start = current_time
                elif current_time - eyes_closed_start >= 2.0:
                    is_paused = not is_paused
                    eyes_closed_start = None
                    winsound.Beep(1000 if not is_paused else 500, 200)
            else:
                eyes_closed_start = None

            ratio_history.append(pitch_ratio)
            smoothed_ratio = float(np.mean(ratio_history))

            yaw_diff = abs(yaw_ratio - yaw_neutral)
            facing_screen = yaw_diff <= yaw_tolerance

            if is_paused:
                status_text = "PAUSE (Yeux 2s pour reprendre)"
                color = (0, 165, 255)
            elif not facing_screen:
                status_text = "BLOQUE (Tete tournee)"
                color = (0, 0, 255)
            else:
                if (direction == 1 and smoothed_ratio > thresh_down) or (direction == -1 and smoothed_ratio < thresh_down):
                    status_text = "SCROLL BAS"
                    color = (0, 255, 0)
                elif (direction == 1 and smoothed_ratio < thresh_up) or (direction == -1 and smoothed_ratio > thresh_up):
                    status_text = "SCROLL HAUT"
                    color = (255, 255, 0)
                else:
                    status_text = "NEUTRE"
                    color = (150, 150, 150)

            x = (w - cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]) // 2
            cv2.putText(display, status_text, (x, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            cv2.putText(display, f"Ratio : {smoothed_ratio:.3f} | Neutre : {r_neutral:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, f"Yaw delta : {yaw_diff:.3f} / Tol : {yaw_tolerance:.2f} ([e] + / [r] -)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if facing_screen else (0, 0, 255), 2)
            cv2.putText(display, f"Seuil BAS : {thresh_down:.3f} ([a] / [d])", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Seuil HAUT : {thresh_up:.3f} ([z] / [s])", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, "[c] Recalibrer | [v] Valider et Sauvegarder | [q] Quitter", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Calibration Head & Gaze', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('c') and state == "RUNNING":
        state = "CALIBRATE_NEUTRAL"
        neutral_ratios.clear()
        down_ratios.clear()
        up_ratios.clear()
        neutral_yaw_ratios.clear()
        ratio_history.clear()
        cv2.setWindowProperty('Calibration Head & Gaze', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        calibration_start_time = time.time()
    
    elif key == ord('a') and state == "RUNNING": thresh_down += 0.01 * direction
    elif key == ord('d') and state == "RUNNING": thresh_down -= 0.01 * direction
    elif key == ord('z') and state == "RUNNING": thresh_up -= 0.01 * direction
    elif key == ord('s') and state == "RUNNING": thresh_up += 0.01 * direction
    
    # Réglage tolérance Yaw (max 0.90)
    elif key == ord('e') and state == "RUNNING": yaw_tolerance = min(0.90, yaw_tolerance + 0.02)
    elif key == ord('r') and state == "RUNNING": yaw_tolerance = max(0.04, yaw_tolerance - 0.02)
    
    elif key == ord('v') and state == "RUNNING":
        config = {
            "type": "head_pose_ratio_advanced",
            "thresh_down": thresh_down,
            "thresh_up": thresh_up,
            "yaw_neutral": yaw_neutral,
            "yaw_tolerance": yaw_tolerance,
            "direction": direction
        }
        with open('config.json', 'w') as f:
            json.dump(config, f)
        print("Paramètres sauvegardés avec succès dans config.json.")
        break

cap.release()
cv2.destroyAllWindows()