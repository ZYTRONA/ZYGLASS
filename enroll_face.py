#!/usr/bin/env python3
"""
enroll_face.py  —  ZYGLASS Social Memory (Mode 5)
====================================================
Captures a face via the laptop webcam, extracts a 128-D face encoding
using the face_recognition library, prompts for the person's name and
saves it to  known_faces/<name>.pkl  so the backend can recognise them
the next time Mode 5 is active.

Usage
-----
    python enroll_face.py

Optional argument:
    python enroll_face.py --name "Alice"   # skip the name prompt

Requirements
------------
    pip install face_recognition opencv-python
"""

import os
import pickle
import sys
import time

# Force X11/XCB backend — avoids Qt Wayland plugin errors on Arch/Wayland
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

try:
    import cv2
except ImportError:
    sys.exit(
        "[ERROR] cv2 not found — you are using the system Python.\n"
        "        Run the script with the project venv:\n\n"
        "            train_env/bin/python enroll_face.py\n"
    )

try:
    import face_recognition  # type: ignore
except ImportError:
    sys.exit(
        "[ERROR] face_recognition is not installed.\n"
        "        Run:  train_env/bin/pip install face_recognition\n"
    )

KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")
WARMUP_FRAMES   = 40        # discard first N frames (auto-exposure stabilises)
CAPTURE_DEVICE  = 0         # 0 = default webcam; change if needed


# ── Helpers ────────────────────────────────────────────────────────────────────

def open_camera(device: int = 0) -> cv2.VideoCapture:
    """Open camera with buffer flush to avoid stale black frames."""
    # Try the requested device first, then auto-scan 0..4
    candidates = [device] + [i for i in range(5) if i != device]
    cap = None
    for idx in candidates:
        # Use default backend (not V4L2) — more reliable for most webcams
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            ok, frame = c.read()
            if ok and frame is not None:
                cap = c
                print(f"[INFO] Opened camera at /dev/video{idx}")
                break
            c.release()
    if cap is None:
        sys.exit("[ERROR] Could not open any camera (tried /dev/video0-4).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Keep buffer size at 1 so we always get the freshest frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def warmup(cap: cv2.VideoCapture, seconds: float = 3.0) -> None:
    """Read and discard frames for `seconds` until a bright frame arrives."""
    print(f"[INFO] Warming up camera (up to {seconds}s)…")
    deadline = time.time() + seconds
    while time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 10:
            print("[INFO] Camera ready.")
            return
        time.sleep(0.05)
    print("[INFO] Warmup timeout — continuing anyway.")


def grab_frame(cap: cv2.VideoCapture):
    """Grab the freshest frame (flush stale buffer frames first)."""
    # Read and discard up to 3 buffered frames, keep the last one
    frame = None
    for _ in range(4):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    return frame


# ── Main ───────────────────────────────────────────────────────────────────────

def enroll(name: str | None = None) -> None:
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    cap = open_camera(CAPTURE_DEVICE)
    warmup(cap)

    print("\n[INFO] Look at the camera and press  SPACE  to capture your face.")
    print("       Press  Q  to quit without saving.\n")

    frame_to_encode = None
    preview_active  = True

    while preview_active:
        frame = grab_frame(cap)
        if frame is None:
            time.sleep(0.03)
            continue

        # Skip frames that are essentially black (camera not ready yet)
        if frame.mean() < 5:
            overlay = frame.copy()
            cv2.putText(overlay, "Camera initialising…", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)
            cv2.imshow("ZYGLASS — Enroll Face", overlay)
            cv2.waitKey(1)
            continue

        # Show live feed so the user can position their face
        display = frame.copy()
        cv2.putText(display, "SPACE = capture  |  Q = quit",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)
        cv2.imshow("ZYGLASS — Enroll Face", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Aborted — no face saved.")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord(' '):
            frame_to_encode = frame.copy()
            print("[INFO] Frame captured — detecting face…")
            break

    cv2.destroyAllWindows()
    cap.release()

    if frame_to_encode is None:
        return

    # ── Detect + encode ────────────────────────────────────────────────────────
    rgb = cv2.cvtColor(frame_to_encode, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")

    if len(locations) == 0:
        print("[ERROR] No face detected in the captured frame.\n"
              "        Make sure your face is clearly visible and try again.")
        return

    if len(locations) > 1:
        print(f"[WARNING] {len(locations)} faces found — using the largest one.")
        # Pick the largest face (by area)
        locations = [max(locations,
                         key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))]

    encodings = face_recognition.face_encodings(rgb, locations)
    if not encodings:
        print("[ERROR] Could not compute face encoding. Try again.")
        return

    encoding = encodings[0]
    print(f"[OK] Face encoding computed ({len(encoding)}-D vector).")

    # ── Ask for name ───────────────────────────────────────────────────────────
    if name is None:
        while True:
            name = input("\nEnter the person's name: ").strip()
            if name:
                break
            print("[WARNING] Name cannot be empty.")

    # ── Save ───────────────────────────────────────────────────────────────────
    safe_name = name.replace(" ", "_")
    out_path  = os.path.join(KNOWN_FACES_DIR, f"{safe_name}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump({"encoding": encoding, "name": name}, f)

    print(f"\n[SUCCESS] Saved face for '{name}'  →  {out_path}")
    print("[INFO] Restart the ZYGLASS backend (backend.py) to load the new face.")

    # Show a confirmation preview
    top, right, bottom, left = locations[0]
    vis = frame_to_encode.copy()
    cv2.rectangle(vis, (left, top), (right, bottom), (0, 255, 120), 2)
    cv2.rectangle(vis, (left, bottom - 30), (right, bottom), (0, 255, 120), -1)
    cv2.putText(vis, name, (left + 6, bottom - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(vis, "Enrolled! Press any key to close.", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 120), 2)
    cv2.imshow("ZYGLASS — Enrolled", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enroll a face for ZYGLASS Social Memory")
    parser.add_argument("--name", type=str, default=None,
                        help="Person's name (skips the interactive prompt)")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    CAPTURE_DEVICE = args.device
    enroll(name=args.name)
