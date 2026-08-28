import cv2
import mediapipe as mp
import time
import numpy as np
import json
from collections import deque

# --- Configuration ---
WINDOW_SIZE = 10
CAMERA_INDEX = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]
face_3d_model = np.array([
    (0.0, 0.0, 0.0),           
    (0.0, -330.0, -65.0),      
    (-225.0, 170.0, -135.0),   
    (225.0, 170.0, -135.0),    
    (-150.0, -150.0, -125.0),  
    (150.0, -150.0, -125.0)    
], dtype=np.float64)

state = "CALIBRATE_NEUTRAL"
calibration_start_time = time.time()
neutral_angles, down_angles = [], []

thresh_down = 0.0
direction = 1
angle_history = deque(maxlen=WINDOW_SIZE)

cap = cv2.VideoCapture(CAMERA_INDEX)
cv2.namedWindow('Calibration Head Pose', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Calibration Head Pose', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
    pitch_angle = None
    face_2d_points = []

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        for idx in LANDMARK_INDICES:
            lm = landmarks[idx]
            face_2d_points.append((lm.x * w, lm.y * h))
        
        face_2d_points = np.array(face_2d_points, dtype=np.float64)
        focal_length = w
        camera_matrix = np.array([[focal_length, 0, h / 2], [0, focal_length, w / 2], [0, 0, 1]], dtype=np.float64)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success_pnp, rot_vec, trans_vec = cv2.solvePnP(face_3d_model, face_2d_points, camera_matrix, dist_matrix)
        
        if success_pnp:
            rmat, _ = cv2.Rodrigues(rot_vec)
            _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(np.matmul(camera_matrix, np.hstack((rmat, trans_vec))))
            pitch_angle = angles[0]

    display = image.copy()
    if len(face_2d_points) > 0:
        for p in face_2d_points:
            cv2.circle(display, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

    if pitch_angle is not None:
        elapsed = current_time - calibration_start_time

        if state == "CALIBRATE_NEUTRAL":
            draw_target(display, w // 2, h // 2)
            cv2.putText(display, "1. Regarde DROIT DEVANT (Tete Droite)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            neutral_angles.append(pitch_angle)
            cv2.putText(display, f"{3.0 - elapsed:.1f}s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            if elapsed > 3.0:
                state, calibration_start_time = "CALIBRATE_DOWN", current_time

        elif state == "CALIBRATE_DOWN":
            draw_target(display, w // 2, h - 50)
            cv2.putText(display, "2. Incline la TETE vers le BAS", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            down_angles.append(pitch_angle)
            cv2.putText(display, f"{3.0 - elapsed:.1f}s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            if elapsed > 3.0:
                a_neutral = np.mean(neutral_angles)
                a_down = np.mean(down_angles)
                
                direction = 1 if a_down > a_neutral else -1
                thresh_down = a_neutral + (a_down - a_neutral) * 0.60
                
                state = "RUNNING"
                cv2.setWindowProperty('Calibration Head Pose', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        elif state == "RUNNING":
            angle_history.append(pitch_angle)
            smoothed_angle = np.mean(angle_history)

            if direction == 1:
                gaze_zone = "ACTION" if smoothed_angle > thresh_down else "NEUTRE"
            else:
                gaze_zone = "ACTION" if smoothed_angle < thresh_down else "NEUTRE"

            color = (0, 255, 0) if gaze_zone == "ACTION" else (0, 0, 255)
            font_s = 1.2 if gaze_zone == "ACTION" else 0.8
            text = "--- ACTION SCROLL ---" if gaze_zone == "ACTION" else "--- NEUTRE ---"
            
            x = (w - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_s, 2)[0][0]) // 2
            cv2.putText(display, text, (x, h // 2), cv2.FONT_HERSHEY_SIMPLEX, font_s, color, 2)

            cv2.putText(display, f"Angle : {smoothed_angle:.1f} | Seuil : {thresh_down:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, "[a] + Difficile | [d] - Difficile", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display, "[v] Valider et Sauvegarder | [r] Recalibrer | [q] Quitter", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Calibration Head Pose', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('r') and state == "RUNNING":
        state, neutral_angles, down_angles = "CALIBRATE_NEUTRAL", [], []
        cv2.setWindowProperty('Calibration Head Pose', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        calibration_start_time = time.time()
        
    elif key == ord('a') and state == "RUNNING": thresh_down += 0.5 * direction  
    elif key == ord('d') and state == "RUNNING": thresh_down -= 0.5 * direction  
    
    elif key == ord('v') and state == "RUNNING":
        config = {"type": "head_pose_down", "thresh_down": thresh_down, "direction": direction}
        with open('config.json', 'w') as f:
            json.dump(config, f)
        print("Paramètres sauvegardés avec succès.")
        break

cap.release()
cv2.destroyAllWindows()