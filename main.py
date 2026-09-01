import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import json
import sys
import time
import winsound
from collections import deque

# Défilement plus fin et direct
SCROLL_STEP_DOWN = -20
SCROLL_STEP_UP = 20
FRAMES_TO_TRIGGER = 1
WINDOW_SIZE = 3
CAMERA_INDEX = 0

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
        thresh_down = config['thresh_down']
        thresh_up = config['thresh_up']
        yaw_neutral = config['yaw_neutral']
        yaw_tolerance = config['yaw_tolerance']
        direction = config['direction']
        
        print("=== AUTO-SCROLL GUITARE (ULTRA-RÉACTIF) ===")
        print(f"Paramètres : Bas={thresh_down:.3f} | Haut={thresh_up:.3f} | Vitesse=20 | Tolérance Yaw={yaw_tolerance:.2f}")
        print("Ferme les yeux 2 s pour Pause / Reprise (bip sonore).")
        print("Appuie sur Ctrl+C pour quitter.")
except FileNotFoundError:
    print("Erreur : config.json introuvable. Lance d'abord 'python test_cam.py' et valide avec [v].")
    sys.exit(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

FACE_TOP, FACE_NOSE, FACE_CHIN = 10, 1, 152
EYE_LEFT_OUTER, EYE_RIGHT_OUTER = 263, 33
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    dist_v1 = np.linalg.norm(pts[1] - pts[5])
    dist_v2 = np.linalg.norm(pts[2] - pts[4])
    dist_h = np.linalg.norm(pts[0] - pts[3])
    return (dist_v1 + dist_v2) / (2.0 * dist_h) if dist_h > 0 else 0.0

ratio_history = deque(maxlen=WINDOW_SIZE)

is_paused = False
eyes_closed_start = None

cap = cv2.VideoCapture(CAMERA_INDEX)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Pause 2s yeux fermés
        ear_left = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (ear_left + ear_right) / 2.0
        eyes_closed = avg_ear < 0.18

        current_time = time.time()
        if eyes_closed:
            if eyes_closed_start is None:
                eyes_closed_start = current_time
            elif current_time - eyes_closed_start >= 2.0:
                is_paused = not is_paused
                eyes_closed_start = None
                winsound.Beep(1000 if not is_paused else 500, 200)
        else:
            eyes_closed_start = None

        if is_paused:
            continue

        # Sécurité orientation Yaw
        x_left = landmarks[EYE_LEFT_OUTER].x
        x_right = landmarks[EYE_RIGHT_OUTER].x
        x_nose = landmarks[FACE_NOSE].x
        eye_dist = abs(x_left - x_right)
        
        if eye_dist > 0:
            yaw_ratio = (x_nose - min(x_left, x_right)) / eye_dist
            yaw_diff = abs(yaw_ratio - yaw_neutral)
            facing_screen = yaw_diff <= yaw_tolerance
        else:
            facing_screen = False

        if not facing_screen:
            continue

        # Ratio vertical Pitch
        y_top = landmarks[FACE_TOP].y
        y_nose = landmarks[FACE_NOSE].y
        y_chin = landmarks[FACE_CHIN].y
        face_height = y_chin - y_top
        
        if face_height > 0:
            pitch_ratio = (y_nose - y_top) / face_height
            ratio_history.append(pitch_ratio)
            smoothed_ratio = float(np.mean(ratio_history))

            # Exécution immédiate
            if (direction == 1 and smoothed_ratio > thresh_down) or (direction == -1 and smoothed_ratio < thresh_down):
                pyautogui.scroll(SCROLL_STEP_DOWN)
            elif (direction == 1 and smoothed_ratio < thresh_up) or (direction == -1 and smoothed_ratio > thresh_up):
                pyautogui.scroll(SCROLL_STEP_UP)

cap.release()