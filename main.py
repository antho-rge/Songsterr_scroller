import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import json
import sys
from collections import deque

# --- Configuration ---
SCROLL_STEP = -20      # Vitesse du défilement (négatif = vers le bas)
FRAMES_TO_TRIGGER = 5  # Nombre d'images d'attente avant de commencer à scroller
WINDOW_SIZE = 12
CAMERA_INDEX = 0

# --- 1. Chargement des paramètres ---
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
        if config.get("type") != "head_pose_ratio":
            print("Attention : Le fichier de config ne correspond pas à la dernière version de calibration.")
        
        thresh_down = config['thresh_down']
        direction = config['direction']
        
        print("=== AUTO-SCROLL GUITARE (Ratio Visage) ===")
        print(f"Configuration chargée (Seuil : {thresh_down:.3f}, Sens : {direction})")
        print("Le script est actif en arrière-plan.")
        print("-> Incline la tête vers le bas pour scroller.")
        print("-> Appuie sur Ctrl+C dans ce terminal pour l'arrêter.")
except FileNotFoundError:
    print("Erreur : Fichier config.json introuvable.")
    print("Lance d'abord 'python test_cam.py' pour calibrer et appuie sur [v] pour sauvegarder.")
    sys.exit(1)

# --- 2. Initialisation ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=False,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

FACE_TOP = 10
FACE_NOSE = 1
FACE_CHIN = 152

ratio_history = deque(maxlen=WINDOW_SIZE)
consecutive_frames = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

# --- 3. Boucle principale invisible ---
while cap.isOpened():
    success, image = cap.read()
    if not success: 
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        y_top = landmarks[FACE_TOP].y
        y_nose = landmarks[FACE_NOSE].y
        y_chin = landmarks[FACE_CHIN].y
        
        face_height = y_chin - y_top
        
        if face_height > 0:
            pitch_ratio = (y_nose - y_top) / face_height
            ratio_history.append(pitch_ratio)
            smoothed_ratio = np.mean(ratio_history)

            # Déclenchement selon la direction configurée
            if direction == 1:
                is_looking_down = smoothed_ratio > thresh_down
            else:
                is_looking_down = smoothed_ratio < thresh_down

            if is_looking_down:
                consecutive_frames += 1
                if consecutive_frames >= FRAMES_TO_TRIGGER:
                    pyautogui.scroll(SCROLL_STEP)
            else:
                consecutive_frames = 0

cap.release()