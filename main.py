import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import json
import sys
from collections import deque

# --- Configuration ---
SCROLL_STEP = -20      # Vitesse du défilement (négatif = vers le bas)
FRAMES_TO_TRIGGER = 8  # Nombre d'images d'attente avant de commencer à scroller
WINDOW_SIZE = 10

# --- 1. Chargement des paramètres ---
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
        thresh_down = config['thresh_down']
        
        print("=== AUTO-SCROLL GUITARE ===")
        print(f"Configuration chargée avec succès (Seuil BAS : {thresh_down:.3f})")
        print("Le script est actif en arrière-plan.")
        print("-> Appuie sur Ctrl+C dans ce terminal pour l'arrêter.")
except FileNotFoundError:
    print("Erreur : Fichier config.json introuvable.")
    print("Lance d'abord 'python test_cam.py' pour calibrer tes yeux et appuie sur [v] pour sauvegarder.")
    sys.exit(1)

# --- 2. Initialisation ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
RIGHT_IRIS_CENTER = 468

ratio_history = deque(maxlen=WINDOW_SIZE)
consecutive_frames = 0

cap = cv2.VideoCapture(0)

# --- 3. Boucle principale invisible ---
while cap.isOpened():
    success, image = cap.read()
    if not success: 
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        y_top = landmarks[RIGHT_EYE_TOP].y
        y_bottom = landmarks[RIGHT_EYE_BOTTOM].y
        y_iris = landmarks[RIGHT_IRIS_CENTER].y
        
        eye_height = y_bottom - y_top
        if eye_height > 0:
            vertical_ratio = (y_iris - y_top) / eye_height
            ratio_history.append(vertical_ratio)
            smoothed_ratio = np.mean(ratio_history)

            # Déclenchement du défilement
            if smoothed_ratio > thresh_down:
                consecutive_frames += 1
                if consecutive_frames >= FRAMES_TO_TRIGGER:
                    pyautogui.scroll(SCROLL_STEP)
            else:
                consecutive_frames = 0

cap.release()