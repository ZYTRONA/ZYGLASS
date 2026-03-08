#!/usr/bin/env python3
"""
Focus State Data Collection Script
Collects facial features (EAR, MAR, Head Pose) and labels them with focus states.

Usage:
    Press F - Record "Focused" state
    Press D - Record "Drowsy" state
    Press A - Record "Looking Away" state
    Press Q - Quit
"""

import cv2
import csv
import numpy as np
import sys

# MediaPipe 0.10.32 uses tasks API (not solutions)
try:
    from mediapipe.tasks import python as mp_python  # type: ignore
    from mediapipe.tasks.python import vision as mp_vision  # type: ignore
    from mediapipe import Image, ImageFormat  # type: ignore
    
    # Initialize FaceLandmarker
    base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')  # type: ignore
    options = mp_vision.FaceLandmarkerOptions(base_options=base_options)  # type: ignore
    face_mesh = mp_vision.FaceLandmarker.create_from_options(options)  # type: ignore
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] New MediaPipe API failed: {e}")
    print("[INFO] Attempting to use OpenCV face detection instead...")
    # Fallback to OpenCV cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # type: ignore
    MEDIAPIPE_AVAILABLE = False

def get_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calc_ear(eye):
    """
    Calculate Eye Aspect Ratio (EAR).
    eye: list of 6 eye landmark points
    Lower EAR indicates closed eyes.
    """
    A = get_distance(eye[1], eye[5])
    B = get_distance(eye[2], eye[4])
    C = get_distance(eye[0], eye[3])
    bottom = np.multiply(2.0, C)
    return (A + B) / bottom

def calc_mar(points):
    """
    Calculate Mouth Aspect Ratio (MAR).
    points: all 468 face landmarks
    Higher MAR indicates open mouth (yawning).
    """
    v_dist = get_distance(points[13], points[14])
    h_dist = get_distance(points[78], points[308])
    return v_dist / h_dist

def main():
    if not MEDIAPIPE_AVAILABLE:
        print("[ERROR] Cannot proceed without MediaPipe")
        sys.exit(1)

    print("=" * 60)
    print("Focus State Data Collection")
    print("=" * 60)
    print("Instructions:")
    print("  F - Record 'Focused' state")
    print("  D - Record 'Drowsy' state")
    print("  A - Record 'Looking Away' state")
    print("  Q - Quit")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    csv_file = 'focus_data.csv'
    try:
        file = open(csv_file, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(['EAR', 'MAR', 'Pitch', 'Yaw', 'Label'])
        print(f"[INFO] Recording data to {csv_file}")
    except IOError as e:
        print(f"[ERROR] Cannot write to {csv_file}: {e}")
        cap.release()
        sys.exit(1)

    frame_count = 0
    label_count = {'Focused': 0, 'Drowsy': 0, 'Looking Away': 0}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[WARNING] Failed to read frame")
            break

        frame_count += 1
        img_h, img_w, _ = frame.shape
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
        except Exception as e:
            print(f"[WARNING] Face detection error: {e}")
            continue

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = []
                for lm in face_landmarks.landmark:
                    x_val = int(np.multiply(lm.x, img_w))
                    y_val = int(np.multiply(lm.y, img_h))
                    points.append([x_val, y_val])

                # Eye landmarks
                left_eye = [points[362], points[385], points[386], points[263], points[374], points[380]]
                right_eye = [points[33], points[159], points[158], points[133], points[153], points[145]]
                
                try:
                    ear = (calc_ear(left_eye) + calc_ear(right_eye)) / 2.0
                    mar = calc_mar(points)
                except Exception as e:
                    print(f"[WARNING] Feature calculation error: {e}")
                    continue

                # Head pose estimation
                face_2d = []
                face_3d = []
                for idx in [33, 263, 1, 61, 291, 199]:
                    lm = face_landmarks.landmark[idx]
                    x_val = int(np.multiply(lm.x, img_w))
                    y_val = int(np.multiply(lm.y, img_h))
                    face_2d.append([x_val, y_val])
                    face_3d.append([x_val, y_val, lm.z])
                
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                
                focal_length = float(img_w)
                cam_matrix = np.array([[focal_length, 0, img_h/2.0], [0, focal_length, img_w/2.0], [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
                try:
                    success_pnp, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(np.hstack((rmat, trans_vec)))
                    
                    pitch = np.multiply(angles[0][0], 360.0)
                    yaw = np.multiply(angles[1][0], 360.0)
                except Exception as e:
                    print(f"[WARNING] Head pose estimation error: {e}")
                    continue

                # Display instructions and current metrics
                cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Pitch: {pitch:.1f} | Yaw: {yaw:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, 'Press F for Focused', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, 'Press D for Drowsy', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, 'Press A for Away', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Collected: F={label_count['Focused']} D={label_count['Drowsy']} A={label_count['Looking Away']}", 
                           (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 200), 2)

        cv2.imshow('Focus Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('f'):
            if results.multi_face_landmarks:
                writer.writerow([ear, mar, pitch, yaw, 'Focused'])
                label_count['Focused'] += 1
                print(f"[SAVED] Focused state #{label_count['Focused']}")
        elif key == ord('d'):
            if results.multi_face_landmarks:
                writer.writerow([ear, mar, pitch, yaw, 'Drowsy'])
                label_count['Drowsy'] += 1
                print(f"[SAVED] Drowsy state #{label_count['Drowsy']}")
        elif key == ord('a'):
            if results.multi_face_landmarks:
                writer.writerow([ear, mar, pitch, yaw, 'Looking Away'])
                label_count['Looking Away'] += 1
                print(f"[SAVED] Looking Away state #{label_count['Looking Away']}")
        elif key == ord('q'):
            print("[INFO] Exiting data collection")
            break

    file.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("Data Collection Summary")
    print("=" * 60)
    print(f"Total samples collected: {sum(label_count.values())}")
    print(f"  Focused: {label_count['Focused']}")
    print(f"  Drowsy: {label_count['Drowsy']}")
    print(f"  Looking Away: {label_count['Looking Away']}")
    print(f"Data saved to: {csv_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
