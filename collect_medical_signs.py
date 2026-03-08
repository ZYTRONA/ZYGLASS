#!/usr/bin/env python3
# cspell:ignore mediapipe pyttsx3 waitKey imshow imencode SRGB landmarker
"""
Medical Sign Language Data Collector  (MediaPipe Tasks API — v0.10.x)
======================================================================
Uses the laptop webcam, detects hand landmarks with the MediaPipe
HandLandmarker Tasks API, and writes 63-coordinate rows to medical_signs.csv.

Prerequisites
-------------
  hand_landmarker.task must exist in the project root.

Usage
-----
  [1] Help   [2] Pain   [3] Yes   [4] No   [5] Water   [q] Quit & save
"""

import csv
import math
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Configuration ────────────────────────────────────────────────────────────
CAMERA_INDEX    = 0          # 0 = default laptop webcam
OUTPUT_CSV      = "medical_signs.csv"
TASK_MODEL_PATH = "hand_landmarker.task"
TARGET_SAMPLES  = 100

SIGNS: dict[int, str] = {
    ord('1'): "Help",
    ord('2'): "Pain",
    ord('3'): "Yes",
    ord('4'): "No",
    ord('5'): "Water",
}
# ─────────────────────────────────────────────────────────────────────────────

# Hand skeleton connections (21 landmarks, index pairs)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (5,9),(9,10),(10,11),(11,12),      # middle
    (9,13),(13,14),(14,15),(15,16),    # ring
    (13,17),(17,18),(18,19),(19,20),   # pinky
    (0,17),(5,9),(9,13),(13,17),       # palm
]


def _draw_hand(frame: np.ndarray, landmarks: list) -> None:
    """Draw hand skeleton directly on a BGR frame."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 180, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 180), -1)


def normalize_landmarks(landmarks: list) -> list[float]:
    """Wrist-centred, scale-invariant normalisation (MediaPipe Tasks API).

    1. Subtract wrist (lm 0)  → hand-centred coordinates.
    2. Divide by wrist→middle-MCP (lm 9) distance → scale-invariant.

    This ensures that hand size and camera distance have no effect on the
    feature vector fed to the classifier.

    Args:
        landmarks: list[NormalizedLandmark] from hand_results.hand_landmarks[0].

    Returns:
        63-element list of normalised floats (21 joints × x, y, z).
    """
    wrist      = landmarks[0]
    middle_mcp = landmarks[9]   # middle finger MCP knuckle

    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    dz = middle_mcp.z - wrist.z
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance == 0:
        distance = 1.0

    norm_row: list[float] = []
    for lm in landmarks:
        norm_row.extend([
            (lm.x - wrist.x) / distance,
            (lm.y - wrist.y) / distance,
            (lm.z - wrist.z) / distance,
        ])
    return norm_row


def _build_csv_headers() -> list[str]:
    """21 landmarks × (x, y, z) + Label column."""
    headers: list[str] = []
    for i in range(21):
        headers.extend([f"x{i}", f"y{i}", f"z{i}"])
    headers.append("Label")
    return headers


def _load_existing_counts(csv_path: str) -> dict[str, int]:
    """Read CSV (if present) and count how many rows exist per label."""
    counts: dict[str, int] = {label: 0 for label in SIGNS.values()}
    if not os.path.exists(csv_path):
        return counts
    try:
        with open(csv_path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                label = row.get("Label", "")
                if label in counts:
                    counts[label] += 1
    except Exception:
        pass
    return counts


def _draw_hud(
    frame: np.ndarray,
    counts: dict[str, int],
    last_saved: str,
    hand_detected: bool,
) -> None:
    """Render a translucent status overlay directly onto frame (in-place)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    panel_height = 30 + len(SIGNS) * 22 + 28
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(
        frame, "AuraHealth AI \u2014 Medical Sign Collector",
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 180), 1,
    )

    key_lookup = {v: chr(k) for k, v in SIGNS.items()}
    y = 42
    for label, count in counts.items():
        filled   = int((count / TARGET_SAMPLES) * 110)
        bar_col  = (0, 255, 100) if count >= TARGET_SAMPLES else (0, 165, 255)
        # Progress bar
        cv2.rectangle(frame, (10, y - 11), (10 + 110, y + 1), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, y - 11), (10 + filled, y + 1), bar_col, -1)
        # Label text
        key_char = key_lookup.get(label, "?")
        cv2.putText(
            frame, f"[{key_char}] {label}: {count}/{TARGET_SAMPLES}",
            (130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, bar_col, 1,
        )
        y += 22

    # Hand detection status indicator
    status_col  = (0, 255, 180) if hand_detected else (0, 60, 255)
    status_text = "\u2713 HAND DETECTED" if hand_detected else "\u2717 NO HAND"
    cv2.putText(frame, status_text, (10, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_col, 1)

    # Flash "SAVED: <label>" in bottom-right for 1.5 s after each save
    if last_saved:
        cv2.putText(
            frame, f"SAVED: {last_saved}",
            (w - 185, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2,
        )


def main() -> None:
    print("=" * 60)
    print("  Medical Sign Language Data Collector")
    print("=" * 60)
    print(f"  Camera  : Laptop webcam (index {CAMERA_INDEX})")
    print(f"  Output  : {OUTPUT_CSV}")
    print(f"  Target  : {TARGET_SAMPLES} samples per sign")
    print()
    for key_ord, label in SIGNS.items():
        print(f"  [{chr(key_ord)}]  →  {label}")
    print("  [q]  →  Quit & save")
    print("=" * 60)

    # ── Verify task model ────────────────────────────────────────────────────
    if not os.path.exists(TASK_MODEL_PATH):
        print(f"\n[ERROR] {TASK_MODEL_PATH} not found in project root.")
        sys.exit(1)

    # ── MediaPipe HandLandmarker (Tasks API) ─────────────────────────────────
    print(f"\n[MEDIAPIPE] Loading {TASK_MODEL_PATH} …")
    base_opts = mp_python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)
    print("[MEDIAPIPE] HandLandmarker ready.")

    # ── CSV (append so existing data is preserved) ───────────────────────────
    file_exists = os.path.exists(OUTPUT_CSV)
    csv_file    = open(OUTPUT_CSV, "a", newline="")
    writer      = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(_build_csv_headers())

    counts = _load_existing_counts(OUTPUT_CSV)

    # ── Laptop webcam ─────────────────────────────────────────────────────────
    print(f"\n[CAMERA] Opening laptop webcam (index {CAMERA_INDEX}) …")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check that no other app is using it.")
        hand_landmarker.close()
        csv_file.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Discard first 40 frames — HP webcam (and most USB cams) start black
    print("[CAMERA] Warming up sensor …")
    for _ in range(40):
        cap.read()

    print("[CAMERA] Webcam ready. Press keys to record samples.\n")

    last_saved_label = ""
    last_saved_time  = 0.0
    hand_row: list[float] = []
    hand_detected         = False
    window_name = "AuraHealth AI \u2014 Sign Collector  (q to quit)"

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # ── Tasks API inference ───────────────────────────────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result    = hand_landmarker.detect(mp_image)

        hand_detected = len(result.hand_landmarks) > 0
        hand_row      = []

        if hand_detected:
            landmarks = result.hand_landmarks[0]  # list[NormalizedLandmark]

            # Draw skeleton directly on the BGR frame
            _draw_hand(frame, landmarks)

            # Normalise: wrist-centred, scaled by wrist→middle-MCP distance
            hand_row = normalize_landmarks(landmarks)

        # ── HUD overlay ──────────────────────────────────────────────────────
        flash = last_saved_label if (time.time() - last_saved_time < 1.5) else ""
        _draw_hud(frame, counts, flash, hand_detected)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # ── Save sample on keypress ───────────────────────────────────────────
        if hand_detected and len(hand_row) == 63 and key in SIGNS:
            label = SIGNS[key]
            writer.writerow(hand_row + [label])
            csv_file.flush()
            counts[label] += 1
            last_saved_label = label
            last_saved_time  = time.time()
            print(f"  [SAVED] {label:<10s}  total: {counts[label]:>3d}/{TARGET_SAMPLES}")
        elif key in SIGNS and not hand_detected:
            print("  [SKIP]  No hand visible \u2014 form the sign clearly and try again.")

    # ── Summary & cleanup ────────────────────────────────────────────────────
    print("\n[DONE] Collection complete.\n")
    all_ok = True
    for label, count in counts.items():
        status = "\u2713" if count >= TARGET_SAMPLES else "\u2717"
        if count < TARGET_SAMPLES:
            all_ok = False
        print(f"  {status}  {label:<12s} {count:>3d} samples")

    if not all_ok:
        print(f"\n[TIP] Some signs have fewer than {TARGET_SAMPLES} samples.")
        print("      Re-run to collect more data.")

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()

    print(f"\n[SAVED]  Data written to {OUTPUT_CSV}")
    print("[NEXT]   Run train_medical_signs.py to build asl_model.pkl")


if __name__ == "__main__":
    main()
