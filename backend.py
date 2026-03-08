# cspell:ignore rapidocr paddleocr googletrans yolov imgsz imencode tobytes pyttsx3
import os
import cv2  # type: ignore
import math
import threading
import time
import tempfile
import json
from typing import Optional
import numpy as np  # type: ignore

# ── Load .env (must be first — sets HF_TOKEN before any HF Hub import) ───────
def _load_dotenv() -> None:
    """Minimal .env loader — no extra dependencies required."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            _v = _v.strip().strip('"').strip("'")  # strip optional quotes
            os.environ.setdefault(_k, _v)           # never overwrite shell env
_load_dotenv()

# Suppress the unauthenticated-request noise when no token is configured
if not os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from ultralytics import YOLO  # type: ignore
from rapidocr_onnxruntime import RapidOCR  # type: ignore
from flask import Flask, Response, jsonify, request  # type: ignore
from flask_cors import CORS  # type: ignore
from googletrans import Translator  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torchvision import models, transforms  # type: ignore
from PIL import Image  # type: ignore
import mediapipe as mp  # type: ignore
from mediapipe.tasks import python as mp_python  # type: ignore
from mediapipe.tasks.python import vision as mp_vision  # type: ignore
import joblib  # type: ignore
import queue
from collections import Counter
import concurrent.futures as _cf

# ── Optional: faster-whisper for offline voice assistant ─────────────────────
try:
    from faster_whisper import WhisperModel  # type: ignore
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[INFO] faster-whisper not installed — Voice Assistant disabled.")
    print("[INFO] Install: pip install faster-whisper")

try:
    import speech_recognition as sr_lib  # type: ignore
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[INFO] SpeechRecognition not installed — Voice Assistant disabled.")

# ── Vosk real-time voice assistant ───────────────────────────────────────────
try:
    import queue
    import sounddevice as sd
    import vosk  # type: ignore
    import json
    import requests  # type: ignore
    from scipy import signal as scipy_signal  # type: ignore
    VOSK_AVAILABLE = True
    print("[VOSK] Real-time voice assistant ready")
except ImportError:
    VOSK_AVAILABLE = False
    print("[INFO] Vosk not installed — using fallback Whisper voice assistant")

try:
    import pyttsx3  # type: ignore
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[INFO] pyttsx3 not available — TTS disabled. Install: pip install pyttsx3")

# ── MongoDB — stores face recognition events and medical sign logs ────────────
try:
    from pymongo import MongoClient  # type: ignore
    from pymongo.errors import ServerSelectionTimeoutError  # type: ignore
    _mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    _mongo_client.admin.command("ping")          # fast liveness check
    _db              = _mongo_client["AIGLASS"]
    db_faces         = _db["face_recognitions"]  # {name, timestamp, confidence}
    db_signs         = _db["medical_signs"]      # {sign, timestamp}
    db_ocr           = _db["ocr_readings"]       # {raw_text, simplified, timestamp}
    MONGO_AVAILABLE  = True
    print("[MONGO] Connected to mongodb://localhost:27017/AIGLASS")
except Exception as _me:
    MONGO_AVAILABLE = False
    db_faces = db_signs = db_ocr = None  # type: ignore
    print(f"[MONGO] MongoDB not reachable ({_me}) — running without database logging.")
    print("[MONGO] Start MongoDB with: sudo systemctl start mongodb")

# ── Ollama — local LLM for medical text simplification (Mode 2 OCR) ──────────
try:
    import ollama as _ollama_lib  # type: ignore
    # Quick connectivity check — list models (fast, no model load)
    _ollama_lib.list()
    OLLAMA_AVAILABLE = True
    OLLAMA_MODEL     = "llama3.2"   # fastest model for single-sentence queries
    print(f"[OLLAMA] Connected — using model '{OLLAMA_MODEL}'")
except Exception as _oe:
    OLLAMA_AVAILABLE = False
    OLLAMA_MODEL     = ""
    print(f"[OLLAMA] Ollama not reachable ({_oe}) — OCR will speak raw text.")
    print("[OLLAMA] Start Ollama with: ollama serve  then  ollama pull llama3.2")
    print("[OLLAMA] ℹ️  This is optional — OCR works without it (just no AI text simplification)")
    print("[OLLAMA] ℹ️  This is optional — OCR works without it (just no AI text simplification)")
try:
    import pyttsx3  # type: ignore
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[INFO] pyttsx3 not available — TTS disabled. Install: pip install pyttsx3")

import pickle

# Face recognition DISABLED for performance optimization
FACE_REC_AVAILABLE = False
print("[INFO] Mode 5 (Face Recognition) DISABLED for performance optimization")

app = Flask(__name__)
CORS(app)

# Mode Configurations - Mode 5 removed for performance
current_mode = 1
modes_info = {
    1: {"name": "Insight Explorer",       "description": "Real-time object detection using YOLOv10 (NMS-free)",            "icon": "🎯",
        "sub_modes": {"currency": {"name": "Currency Reader",   "description": "Indian Rupee banknote detection + spoken denomination", "icon": "💵"}}},
    2: {"name": "Multilingual Reader",     "description": "OCR with live translation",                                    "icon": "📖"},
    3: {"name": "Emotion Analyzer",        "description": "Facial emotion recognition",                                    "icon": "😊"},
    4: {"name": "Medical Sign Translator", "description": "Hand sign language → spoken medical words via Random Forest",    "icon": "🤝"}
}

# Mode Statistics - Mode 5 and lecture mode removed
mode_stats = {
    1: {"objects_detected": 0, "fps": 0,
        "currency_mode": False, "notes_detected": 0, "last_denomination": "None"},
    2: {"texts_recognized": 0, "translations": 0},
    3: {"faces_detected": 0, "emotions": {}},
    4: {"last_sign": "None", "signs_detected": 0}
}

# ── Telemetry tracking for /telemetry endpoint ──────────────────────────────
_telemetry_lock = threading.Lock()
_stream_fps: list = [0.0]           # [current FPS]
_last_frame_latency: list = [0.0]   # [last frame encode time in ms]
_frame_times: list = [[]]           # [list of recent frame timestamps]
_esp32_connected: list = [True]     # [ESP32 connection status]
_event_log: list = []               # [{time, type, message, color}]
EVENT_LOG_MAX = 100                 # max events in memory

# ── Training Data Collection Enhancement ────────────────────────────────────
def save_training_sample(mode_name: str, frame: np.ndarray, annotation: str = "") -> str:
    """Save frame as training sample with timestamp and mode context."""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # millisecond precision
        filename = f"training_{mode_name}_{timestamp}.jpg"
        filepath = f"datasets/training_samples/{filename}"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save frame with high quality for training
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Log annotation if provided
        if annotation:
            log_file = f"datasets/training_samples/annotations_{mode_name}.txt"
            with open(log_file, "a") as f:
                f.write(f"{filename}\t{annotation}\t{timestamp}\n")
        
        print(f"[TRAINING] Saved: {filename}")
        return filepath
    except Exception as e:
        print(f"[TRAINING] Save failed: {e}")
        return ""

def enhance_frame_for_training(frame: np.ndarray) -> np.ndarray:
    """Apply enhancement filters for better training data quality."""
    try:
        # Denoise while preserving edges
        enhanced = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Enhance contrast and brightness slightly
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)
        
        # Sharpen slightly for better feature definition
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    except Exception:
        return frame  # Return original on any error

# ── Voice Assistant state (shared between voice_worker and API endpoints) ─────
_voice_lock = threading.Lock()
_voice_state: dict = {
    "enabled": False,
    "listening": False,
    "last_command": "",
    "last_response": "",
    "wake_word_fired": False,
}

# Thread-safe mode switching and frame handling with queue-based architecture
mode_lock = threading.RLock()
latest_ocr_data = []
text_for_ui = []
current_emotion = "Detecting"
detected_faces = 0
frame_skip_counter = 0
frame_skip_rate = 1  # Process every frame (60fps capable with multithreading)
emo_inference_counter = 0  # Skip emotion inference frames
frame_timestamp = 0  # Track frame age to detect stale frames
FRAME_TIMEOUT = 2.0  # Drop frames older than 2 s (ESP32 WiFi can spike > 500 ms)

# Thread-safe queues for frame pipeline
# capture_queue: feeds frames from camera capture to AI processing
# display_queue: feeds processed frames from AI to web serving
capture_queue = queue.Queue(maxsize=1)  # Only ever keep the single freshest frame
display_queue = queue.Queue(maxsize=1)  # Always serve freshest annotated frame (min latency)
pipeline_running = True

# ── Mode 2 OCR state — written only by ocr_worker, read by AI thread & /ocr_data ──
ocr_lock       = threading.Lock()
ocr_results: list[dict] = []       # [{text, translation, confidence, tl, br}]
_ocr_src_frame: list     = [None, 0.0]  # [frame, ts] fed from AI thread to OCR worker
_ocr_src_lock  = threading.Lock()
OCR_INTERVAL   = 1.5               # seconds between PaddleOCR (RapidOCR-ONNX) passes

# ── Currency Reader sub-mode (Mode 1) ─────────────────────────────────────────
currency_mode: bool   = False          # toggled via /toggle_currency
_currency_src_frame:  list = [None, 0.0]
_currency_src_lock    = threading.Lock()
_currency_boxes:      list = []        # [{denomination, conf, tl, br}]
_currency_boxes_lock  = threading.Lock()
_currency_last_spoken: list  = [""]   # debounce — don't repeat same denom
_currency_last_time:   list  = [0.0]
CURRENCY_TTS_COOLDOWN = 4.0           # seconds before repeating same denomination

# Load rupee YOLO model (optional — system works without it, shows warning in Mode 1)
rupee_model       = None
RUPEE_MODEL_AVAILABLE = False
for _rp in ('rupee_model.pt', 'rupee_model.onnx', 'rupee_model.engine'):
    if os.path.exists(_rp):
        try:
            rupee_model = YOLO(_rp, task='detect')
            RUPEE_MODEL_AVAILABLE = True
            print(f"[SUCCESS] Loaded rupee detection model: {_rp}")
            break
        except Exception as _re:
            print(f"[WARNING] Could not load {_rp}: {_re}")
if not RUPEE_MODEL_AVAILABLE:
    print("[INFO] No rupee_model found — Currency Reader disabled.")
    print("[INFO] Run train_rupee_yolo.py to train it, then restart backend.")

# ── Lecture Logger sub-mode (Mode 2) ──────────────────────────────────────────
lecture_mode: bool    = False          # toggled via /toggle_lecture
_lecture_src_frame:   list = [None, 0.0]
_lecture_src_lock     = threading.Lock()
_lecture_lock         = threading.Lock()
_lecture_lines:       list = []        # [{text, timestamp}]  rolling whiteboard log
_lecture_summary:     list = [""]      # [current AI summary]
_lecture_last_raw:    list = [""]      # last combined raw text sent to LLM
_lecture_pending:     list = [False]   # LLM in-flight flag
LECTURE_OCR_INTERVAL   = 1.5          # seconds between whiteboard OCR passes
LECTURE_SUMMARIZE_EVERY = 30.0        # seconds between full summarizations
_lecture_last_summary_ts: list = [0.0]

# ── Mode 1 YOLO worker state (runs inference in bg, AI thread only draws) ─────
_yolo_src_frame:  list = [None, 0.0]
_yolo_src_lock    = threading.Lock()
_yolo_boxes:      list = []    # [{label, tl, br}]
_yolo_boxes_lock  = threading.Lock()

# ── Mode 3 Emotion worker state ───────────────────────────────────────────────
_emotion_src_frame: list = [None, 0.0]
_emotion_src_lock   = threading.Lock()
_emotion_cache: dict = {"boxes": [], "emotion": "Detecting"}
_emotion_cache_lock = threading.Lock()
_emotion_history: list = []   # rolling window of last 5 raw predictions (anti-flicker)

# ── Mode 4 Sign worker state (bg MediaPipe inference, AI thread only draws) ───
_sign_src_frame:  list = [None, 0.0]
_sign_src_lock    = threading.Lock()
_sign_cache: dict = {"sign": "No Hand Detected", "landmarks": None}
_sign_cache_lock  = threading.Lock()

# ── Mode 5 Face worker state (bg scan, AI thread only draws cached boxes) ─────
_face_worker_frame: list = [None, 0.0]
_face_worker_lock   = threading.Lock()
FACE_SCAN_INTERVAL  = 1.0   # HOG+encoding is heavy — 1 Hz is plenty for face tracking

# ── FPS tracker (counts annotated frames pushed per second) ───────────────────
_fps_times: list = []   # timestamps of recent displayed frames (AI thread only)

# MediaPipe Setup for Mode 4 (Medical Sign Translator)
# sign_history is appended in the AI thread and read in the /sign_data endpoint.
sign_history: list = []
_sign_votes:  list = []   # rolling raw predictions for temporal majority vote
VOTE_WINDOW      = 9      # look-back window (frames)
VOTE_THRESHOLD   = 5      # agreements needed out of VOTE_WINDOW to confirm a sign
_TASK_MODEL_PATH = "hand_landmarker.task"
_LM_SPEC   = mp_vision.drawing_utils.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=3)
_CON_SPEC  = mp_vision.drawing_utils.DrawingSpec(color=(0, 180, 255), thickness=2)
_HAND_CONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS

try:
    if not os.path.exists(_TASK_MODEL_PATH):
        raise FileNotFoundError(f"{_TASK_MODEL_PATH} not found")
    _base = mp_python.BaseOptions(model_asset_path=_TASK_MODEL_PATH)
    _opts = mp_vision.HandLandmarkerOptions(
        base_options=_base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    mp_hands_detector = mp_vision.HandLandmarker.create_from_options(_opts)
    HANDS_AVAILABLE = True
    print("[SUCCESS] MediaPipe HandLandmarker initialized for Mode 4")
except Exception as e:
    print(f"[WARNING] HandLandmarker init failed: {e}")
    print("[INFO] Download hand_landmarker.task and place it in the project root.")
    mp_hands_detector = None
    HANDS_AVAILABLE = False

import urllib.request as _urllib_req

class VideoStream:
    """
    MJPEG stream reader for ESP32-CAM.
    Optimised for minimum latency:
      • 4 KB reads — frame arrives as soon as network delivers it, not after 64 KB fills
      • Drain-to-latest: when the buffer holds multiple JPEGs, only the newest is kept;
        this prevents the “sliding backlog” that made the feed look like a delayed recording
      • threading.Event so generate_frames blocks at zero CPU until a new frame is ready;
        no busy-poll, no duplicate frames sent to the browser
    """
    _BOUNDARY = b'--'  # MJPEG part boundary prefix (Content-Length parsing)

    def __init__(self, url: str) -> None:
        self.url        = url
        self.raw_jpeg: Optional[bytes]   = None  # latest raw JPEG bytes from ESP32
        self.frame:    Optional[np.ndarray] = None  # kept for backward compat (unused)
        self.timestamp  = 0.0
        self.frame_num  = 0                # increments on every new frame
        self.stopped    = False
        self.lock       = threading.Lock()
        self._new_frame = threading.Event()   # set every time a decoded frame is stored
        self._stream    = None
        self._buf       = b''
        self._connect()

    def _connect(self) -> None:
        try:
            self._stream = _urllib_req.urlopen(self.url, timeout=10)
            self._buf    = b''
            print(f"[STREAM] Connected to {self.url}")
        except Exception as e:
            # Silently fail - camera is optional for voice-only mode
            self._stream = None

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    # ── Stream helpers ──────────────────────────────────────────────────────
    def _refill(self, need: int = 512) -> None:
        """Append bytes from the network until buf has at least `need` bytes."""
        while len(self._buf) < need:
            chunk = self._stream.read(max(4096, need - len(self._buf)))  # type: ignore[union-attr]
            if not chunk:
                raise ConnectionError("stream closed")
            self._buf += chunk

    def _read_line(self) -> bytes:
        """Read one CRLF-terminated line; fills buffer as needed.
        1024-byte reads for faster WiFi error detection on poor connections.
        """
        while True:
            idx = self._buf.find(b'\r\n')
            if idx != -1:
                line = self._buf[:idx]
                self._buf = self._buf[idx + 2:]
                return line
            chunk = self._stream.read(1024)  # type: ignore[union-attr]
            if not chunk:
                raise ConnectionError("stream closed")
            self._buf += chunk

    def _read_exactly(self, n: int) -> bytes:
        """Read exactly n bytes: drain the internal buffer first, then pull the
        remainder straight from the socket in large chunks.  Much faster than
        refill() which grew the internal buffer to n bytes before slicing."""
        chunks: list = []
        remaining = n
        # 1. Consume whatever is already buffered (no copy needed)
        if self._buf:
            take = min(len(self._buf), remaining)
            chunks.append(self._buf[:take])
            self._buf = self._buf[take:]
            remaining -= take
        # 2. Read the rest directly from the socket
        while remaining > 0:
            chunk = self._stream.read(min(remaining, 32768))  # type: ignore[union-attr]
            if not chunk:
                raise ConnectionError("stream closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def update(self) -> None:
        while not self.stopped:
            if self._stream is None:
                time.sleep(0.1)   # fast reconnect for WiFi stability
                self._connect()
                continue
            try:
                # ── 1. Locate the MJPEG boundary line ─────────────────────────
                line = self._read_line()
                if self._BOUNDARY not in line:
                    continue  # skip preamble / empty lines

                # ── 2. Parse headers — extract Content-Length ──────────────────
                content_length = 0
                while True:
                    header = self._read_line()
                    if not header:
                        break  # blank line = end of headers
                    if header.lower().startswith(b'content-length:'):
                        try:
                            content_length = int(header.split(b':', 1)[1].strip())
                        except ValueError:
                            pass

                if content_length <= 0:
                    continue  # malformed frame — skip

                # ── 3. Read exactly content_length bytes → store raw JPEG ───────
                jpg_bytes = self._read_exactly(content_length)
                if jpg_bytes:
                    with self.lock:
                        self.raw_jpeg  = jpg_bytes
                        self.timestamp = time.time()
                        self.frame_num += 1
                        self._new_frame.set()   # signal INSIDE lock → no race

            except Exception as e:
                print(f"[STREAM] Read error: {e} — reconnecting…")
                self._stream = None
                self._buf    = b''

    def read(self) -> Optional[np.ndarray]:
        """Decode the latest raw JPEG for AI worker threads (on demand)."""
        with self.lock:
            raw = self.raw_jpeg
            ts  = self.timestamp
        if raw is None or time.time() - ts >= FRAME_TIMEOUT:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def wait_new_frame(self, timeout: float = 0.5) -> tuple:
        """Block until a new JPEG arrives (or timeout). Returns (raw_jpeg, timestamp).
        Zero CPU wait — wakes immediately when update() stores a new frame.
        Returns (None, 0.0) only on timeout (no new frame within timeout period).
        """
        # Snapshot the current frame_num before waiting
        with self.lock:
            initial_frame_num = self.frame_num
        
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None, 0.0   # timeout — no new frame arrived
            
            got_event = self._new_frame.wait(timeout=remaining)
            self._new_frame.clear()
            
            with self.lock:
                # Check if frame_num actually incremented (genuine new frame)
                if self.frame_num > initial_frame_num:
                    return self.raw_jpeg, self.timestamp
            
            # Spurious wakeup or old event — retry until timeout

    def is_frame_fresh(self) -> bool:
        with self.lock:
            return time.time() - self.timestamp < FRAME_TIMEOUT

    def stop(self):
        self.stopped = True
        if self._stream:
            self._stream.close()

print("Loading AI Models...")

# Load YOLO model — priority: TensorRT engine > ONNX > PyTorch weights > yolov8n fallback
# yolov10n.onnx was exported successfully (8.9 MB, opset 20) — use it when TRT unavailable.
# TRT background export is skipped: tensorrt-cu12 is not installed on this system.
try:
    model = YOLO('yolov10n.engine', task='detect')  # TensorRT — fastest (requires tensorrt-cu12)
    print("[SUCCESS] Loaded TensorRT yolov10n engine")
except Exception:
    try:
        model = YOLO('yolov10n.onnx', task='detect')  # ONNX runtime — no TRT/CUDA needed, ~2× faster than .pt
        print("[SUCCESS] Loaded yolov10n.onnx (ONNX runtime)")
    except Exception:
        try:
            model = YOLO('yolov10n.pt', task='detect')  # Plain PyTorch weights
            print("[INFO] Loaded yolov10n.pt (PyTorch)")
        except Exception:
            model = YOLO('yolov8n.pt', task='detect')   # Ultimate fallback
            print("[FALLBACK] yolov10n not found — using yolov8n.pt")

ocr_engine = RapidOCR()   # PaddleOCR ONNX — handles EN/JA, no framework needed
translator = Translator()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = models.resnet50(weights=None)
num_ftrs = emotion_model.fc.in_features
emotion_model.fc = nn.Sequential(  # type: ignore
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 7)
)

# Load the trained emotion model checkpoint
checkpoint = torch.load('runs/emotion_model_v1/emotion_model_best.pth', map_location=device)
emotion_model.load_state_dict(checkpoint['model_state'])
emotion_model = emotion_model.to(device)
emotion_model.eval()

# Compile emotion model to TorchScript for ~25% faster inference
_TS_PATH = 'runs/emotion_model_v1/emotion_model.ts'
try:
    if os.path.exists(_TS_PATH):
        emotion_model = torch.jit.load(_TS_PATH, map_location=device)  # type: ignore
        print("[SUCCESS] Loaded TorchScript emotion model (cached)")
    else:
        with torch.no_grad():
            _dummy = torch.randn(1, 3, 48, 48).to(device)
            emotion_model = torch.jit.trace(emotion_model, _dummy)  # type: ignore
        torch.jit.save(emotion_model, _TS_PATH)  # type: ignore
        print("[SUCCESS] Emotion model compiled to TorchScript and cached")
except Exception as _ts_err:
    print(f"[INFO] TorchScript compile skipped: {_ts_err} — using standard PyTorch")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # type: ignore

# ── _WrappedModel: must be defined here so pickle can find __main__._WrappedModel
# when joblib deserialises asl_model.pkl (class is identical to train_medical_signs.py)
class _WrappedModel:
    """Thin sklearn wrapper stored inside asl_model.pkl.
    Must be defined in backend.py (== __main__) so joblib/pickle can reconstruct it.
    """
    def __init__(self, ensemble, le) -> None:
        self._ensemble = ensemble
        self._le       = le

    def predict(self, X) -> np.ndarray:
        encoded = self._ensemble.predict(np.asarray(X, dtype=np.float32))
        return self._le.inverse_transform(encoded)

    def predict_proba(self, X) -> np.ndarray:
        return self._ensemble.predict_proba(np.asarray(X, dtype=np.float32))

    def __repr__(self) -> str:
        return f"_WrappedModel(ensemble={self._ensemble!r})"


# Load ASL Medical Sign Model for Mode 4
asl_model = None
asl_model_available = False
if os.path.exists('asl_model.pkl'):
    try:
        asl_model = joblib.load('asl_model.pkl')
        asl_model_available = True
        print("[SUCCESS] Loaded asl_model.pkl for Mode 4 (Medical Sign Translator)")
    except Exception as e:
        print(f"[WARNING] Failed to load asl_model.pkl: {e}")
else:
    print("[INFO] asl_model.pkl not found — run collect_medical_signs.py then train_medical_signs.py")
    print("[INFO] Mode 4 will display 'Model Not Loaded' until asl_model.pkl is present.")

# ── Mode 5 Social Memory ── face recognition state ───────────────────────────
known_face_encodings: list = []
known_face_names: list = []
face_history: list = []           # rolling history [{name, time}], max 20 entries
_face_locations: list = []        # cached between frames (updated every N frames)
_face_names: list = []            # cached between frames
_last_mode5_frame: list = [None]  # most recent raw frame — used by /enroll_unknown

if not os.path.exists("known_faces"):
    os.makedirs("known_faces")
    print("[INFO] Created known_faces/ — run enroll_face.py to add people")

for _fn in sorted(os.listdir("known_faces")):
    if _fn.endswith(".pkl"):
        try:
            with open(os.path.join("known_faces", _fn), "rb") as _f:
                _data = pickle.load(_f)
                # Support both old format {encoding, name} and new {encodings: [...], name}
                if "encodings" in _data:
                    for _enc in _data["encodings"]:
                        known_face_encodings.append(_enc)
                        known_face_names.append(_data["name"])
                else:
                    known_face_encodings.append(_data["encoding"])
                    known_face_names.append(_data["name"])
        except Exception as _e:
            print(f"[WARNING] Could not load {_fn}: {_e}")

print(f"[SUCCESS] Mode 5: loaded {len(known_face_names)} known face(s): {known_face_names}")

# TTS debounce state — stored in mutable lists so threads can mutate without 'global'


def normalize_landmarks(landmarks) -> list:
    """Wrist-centred, scale-invariant normalisation for MediaPipe Tasks API.

    Identical algorithm to collect_medical_signs.py so inference features
    exactly match the training data:
      1. Subtract wrist (lm 0)  → hand-centred coordinates.
      2. Divide by wrist→middle-MCP (lm 9) distance → scale-invariant.

    Args:
        landmarks: list[NormalizedLandmark] from hand_results.hand_landmarks[0].

    Returns:
        63-element list of normalised floats.
    """
    wrist      = landmarks[0]
    middle_mcp = landmarks[9]   # middle finger MCP knuckle
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    dz = middle_mcp.z - wrist.z
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance == 0:
        distance = 1.0
    norm_row: list = []
    for lm in landmarks:
        norm_row.extend([
            (lm.x - wrist.x) / distance,
            (lm.y - wrist.y) / distance,
            (lm.z - wrist.z) / distance,
        ])
    return norm_row


def _normalize_landmarks(raw_63: list) -> list:
    """Legacy flat-list normalisation (kept for backward compat with old pkl files).
    Prefer normalize_landmarks() for new inference paths.
    """
    pts = np.array(raw_63, dtype=np.float32).reshape(21, 3)
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[9]))     # wrist → middle-MCP distance (lm 9)
    if scale > 1e-6:
        pts /= scale
    return pts.flatten().tolist()


# ── Audio Alert System ───────────────────────────────────────────────────────
class AudioAlertSystem:
    """Background TTS queue — clears stale alerts so only the latest speaks."""
    def __init__(self) -> None:
        self.speech_queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()

    def _process_queue(self) -> None:
        if not TTS_AVAILABLE:
            return
        try:
            engine = pyttsx3.init()  # type: ignore
            # ── Voice clarity settings (Female voice preference) ─────────────
            engine.setProperty('rate', 160)   # Slightly faster for female voice
            engine.setProperty('volume', 0.9) # Clear but not too loud
            
            # Pick a female English voice if available
            try:
                voices = engine.getProperty('voices')
                if not voices:
                    voices = []
                # Don't try to convert voices, work with whatever we get
                elif not isinstance(voices, list):
                    # Just work with the object as-is, handle iteration errors later
                    pass
            except Exception:
                voices = []
            
            female_voice = None
            
            # Look for female voices first
            if voices:
                try:
                    # Safe iteration over voices (could be list or other iterable)
                    for voice in voices:  # type: ignore[misc]
                        try:
                            voice_name = (getattr(voice, 'name', '') or '').lower()
                            voice_id = (getattr(voice, 'id', '') or '').lower()
                            if any(keyword in voice_name for keyword in ['female', 'woman', 'girl', 'zira', 'hazel', 'karen', 'susan', 'samantha']):
                                if 'en' in voice_id or 'english' in voice_name:
                                    female_voice = voice
                                    break
                        except Exception:
                            continue
                except (TypeError, AttributeError):
                    # voices is not iterable, skip female voice detection
                    pass
            
            if female_voice:
                engine.setProperty('voice', female_voice.id)
                print(f"[AUDIO] Female Voice: {getattr(female_voice, 'name', 'Unknown')} | rate=160 | volume=0.9")
            else:
                # Fallback to regular English voice selection
                if voices:
                    try:
                        en_voices = []
                        # Safe iteration over voices
                        for v in voices:  # type: ignore[misc]
                            try:
                                v_id = (getattr(v, 'id', '') or '').lower()
                                v_name = (getattr(v, 'name', '') or '').lower()
                                if 'en' in v_id or 'english' in v_name:
                                    en_voices.append(v)
                            except Exception:
                                continue
                    except (TypeError, AttributeError):
                        # voices is not iterable, use default
                        en_voices = []
                        
                    if en_voices:
                        preferred = None
                        for v in en_voices:
                            v_id = (getattr(v, 'id', '') or '').lower()
                            if 'us' in v_id or 'gb' in v_id:
                                preferred = v
                                break
                        if not preferred:
                            preferred = en_voices[0]
                            
                        engine.setProperty('voice', preferred.id)
                        print(f"[AUDIO] Voice: {getattr(preferred, 'name', 'Unknown')} | rate=160 | volume=0.9")
                    else:
                        print("[AUDIO] No English voice found — using system default")
                else:
                    print("[AUDIO] No voices available — using system default")
        except Exception as exc:
            print(f"[AUDIO] Engine init failed: {exc}")
            return
        while True:
            try:
                text = self.speech_queue.get(timeout=1.0)
                if text:
                    engine.say(text)
                    engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as exc:
                print(f"[AUDIO] Speak error: {exc}")

    def speak(self, text: str) -> None:
        """Enqueue text, clearing any waiting utterance first."""
        if not TTS_AVAILABLE or not text:
            return
        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()
        self.speech_queue.put(text)

audio = AudioAlertSystem()

# ───────────────────────────────────────────────────────────────
# Offline Voice Assistant
# ───────────────────────────────────────────────────────────────
# Shared voice state — polled by /voice_data every 500 ms from React UI
_voice_state: dict = {
    "active":        False,   # True while recording + processing
    "listening":     False,   # True while mic VAD is waiting for speech
    "wake_detected": False,   # True for 2 s after wake word fires
    "last_command":  "",      # last transcribed command
    "last_response": "",      # last spoken response
    "last_ts":       "",      # HH:MM:SS of last command
    "enabled":       False,   # toggled via /toggle_voice
}
_voice_lock = threading.Lock()


class FastVoiceAssistant:
    """
    Real-time Vosk-powered voice assistant with instant wake word detection.
    
    Uses "assistant" as wake word — guaranteed dictionary word, no phonetic guessing.
    Processes audio in real-time with zero buffering lag, unlike Whisper's batch approach.
    
    Architecture:
      sounddevice → 16kHz PCM chunks → Vosk KaldiRecognizer → JSON parse →
      wake word check → Ollama LLM → AudioAlertSystem TTS
    
    Latency: ~50-200ms from speech to response (vs 2-10s for Whisper pipeline)
    """
    
    def __init__(self, audio_engine) -> None:
        self.q = queue.Queue()
        self.conversational_mode = False  # Stay active after wake word
        self.last_activity_time = 0
        self.conversation_timeout = 30  # seconds of inactivity before auto-exit
        self.device_samplerate = 16000  # Will be set in listen_loop
        self.resample_needed = False
        
        try:
            if not os.path.exists("model"):
                raise FileNotFoundError("Vosk model directory not found. Download vosk-model-small-en-us-0.15")
            self.model = vosk.Model("model")
            print("[VOSK] Model loaded successfully")
        except Exception as e:
            print(f"[VOSK] Model load failed: {e} — voice assistant disabled")
            self._enabled = False
            return
        
        self.audio_engine = audio_engine
        self._enabled = True
        
        # Voice state for React UI
        with _voice_lock:
            _voice_state["enabled"] = True
            _voice_state["listening"] = True
        
        self.thread = threading.Thread(target=self.listen_loop, daemon=True, name="VoskVoice")
        self.thread.start()
        print("[VOSK] Assistant ready — say 'hey zyglass' to start conversation ✓")
    
    def audio_callback(self, indata, frames, time, status):
        """Sounddevice callback — pushes raw PCM to queue for Vosk processing."""
        if status:
            print(f"[VOSK] Audio status: {status}")
        if self._enabled:
            # Resample if needed (Bluetooth devices often use 48kHz, Vosk needs 16kHz)
            if self.resample_needed and self.device_samplerate != 16000:
                # Convert bytes to numpy array
                audio_data = np.frombuffer(indata, dtype=np.int16)
                # Resample from device rate to 16kHz
                num_samples = int(len(audio_data) * 16000 / self.device_samplerate)
                resampled = scipy_signal.resample(audio_data, num_samples)
                # scipy.signal.resample can return a tuple, extract the array
                if isinstance(resampled, tuple):
                    resampled = resampled[0]
                # Convert back to int16 bytes
                resampled_bytes = np.asarray(resampled, dtype=np.int16).tobytes()
                self.q.put(resampled_bytes)
            else:
                self.q.put(bytes(indata))
            
            # Debug: print every 10 callbacks initially, then every 50
            if not hasattr(self, '_callback_count'):
                self._callback_count = 0
                print("[VOSK] ✓ Audio callback is being called!")
            self._callback_count += 1
            
            # More frequent updates at start to confirm it's working
            if self._callback_count <= 100:
                if self._callback_count % 10 == 0:
                    print(f"[VOSK] Audio flowing... ({self._callback_count} callbacks)")
            elif self._callback_count % 50 == 0:
                print(f"[VOSK] Audio active... ({self._callback_count} callbacks) - Say 'Hey ZyGlass'")
    
    def listen_loop(self):
        """Main recognition loop — real-time Vosk transcription + wake word detection."""
        try:
            # Find AirPods or best input device
            devices = sd.query_devices()
            input_device = None
            device_samplerate = 16000
            
            # Use default microphone (Bluetooth not working in PulseAudio)
            input_device = sd.default.device[0]
            device_info = sd.query_devices(input_device, 'input')
            device_samplerate = int(device_info['default_samplerate'])
            print(f"[VOSK] Using built-in microphone: {device_info['name']}")
            print(f"[VOSK] (Bluetooth devices not configured for input in PulseAudio)")
            
            print(f"[VOSK] Device sample rate: {device_samplerate} Hz")
            if device_samplerate != 16000:
                print(f"[VOSK] Will resample {device_samplerate} Hz → 16000 Hz for Vosk")
                self.resample_needed = True
            
            self.device_samplerate = device_samplerate
            
            print(f"[VOSK] Opening audio stream with device {input_device}...")
            # Use device native sample rate
            with sd.RawInputStream(
                samplerate=device_samplerate, blocksize=int(device_samplerate * 0.5), 
                dtype='int16', channels=1, callback=self.audio_callback, device=input_device
            ):
                print("[VOSK] ✓ Audio stream opened successfully!")
                rec = vosk.KaldiRecognizer(self.model, 16000)
                print("[VOSK] Listening for 'hey zyglass' wake word... (conversational mode available)")
                print("[VOSK] Audio stream started - speak now!")
                print("[VOSK] Waiting for audio callbacks...")
                
                while self._enabled:
                    try:
                        data = self.q.get(timeout=1.0)
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            text = result.get("text", "").strip()
                            
                            if text:  # Only log non-empty transcriptions
                                print(f"[VOSK] Heard: '{text}'")
                            
                            # Check for conversational mode commands
                            if self.conversational_mode:
                                # Check for exit commands
                                if any(word in text.lower() for word in ["stop listening", "goodbye", "bye", "exit", "turn off"]):
                                    self.conversational_mode = False
                                    print("[VOSK] ✓ Exiting conversational mode")
                                    self._respond("Okay, I'll stop listening now. Say hey zyglass to wake me up again.")
                                    continue
                                
                                # Process any command in conversational mode
                                if text.strip():
                                    self.last_activity_time = time.time()
                                    print(f"[VOSK] ✓ CONVERSATIONAL COMMAND: '{text}'")
                                    
                                    # Route to fast path or Ollama
                                    if self._route_fast(text.strip()):
                                        pass  # handled by fast path
                                    else:
                                        # Send to Ollama for general AI responses
                                        threading.Thread(
                                            target=self.ask_llama, args=(text.strip(),), daemon=True
                                        ).start()
                                continue
                            
                            # Check for wake word (only when not in conversational mode)
                            if not self.conversational_mode:
                                wake_detected = False
                                command = ""
                                text_lower = text.lower().strip()
                                
                                # RELAXED wake word detection - accepts partial matches
                                wake_variants = [
                                    "hey zyglass", "zyglass", "hey zy glass", "zy glass",
                                    "hey ziglass", "ziglass", "hey zi glass", "zi glass",
                                    "hey glass", "glass", "hey zig", "zig",
                                    "hey z", "hey zee", "zee glass", "see glass",
                                    "hey sai", "sai glass", "sy glass",
                                    # Common misrecognitions
                                    "hey guys", "guys", "hey gas", "hey class", "hey gus"
                                ]
                                
                                # Check if ANY variant appears in the text (fuzzy matching)
                                for variant in wake_variants:
                                    if variant in text_lower:
                                        wake_detected = True
                                        # Extract command after the wake phrase
                                        idx = text_lower.find(variant)
                                        if idx + len(variant) < len(text_lower):
                                            command = text_lower[idx + len(variant):].strip()
                                        break
                            
                                if wake_detected:
                                    print(f"[VOSK] ✓ WAKE WORD DETECTED! Entering conversational mode")
                                    self.conversational_mode = True
                                    self.last_activity_time = time.time()
                                    
                                    # Update voice state for React UI
                                    with _voice_lock:
                                        _voice_state["wake_detected"] = True
                                        _voice_state["last_command"] = command or "(wake word only)"
                                        _voice_state["last_ts"] = time.strftime("%H:%M:%S")
                                        _voice_state["active"] = True
                                    
                                    # Clear wake indicator after 2s
                                    threading.Timer(2.0, self._clear_wake).start()
                                    
                                    if command:
                                        # Route to fast path or Ollama
                                        if self._route_fast(command):
                                            pass  # handled by fast path
                                        else:
                                            # Send to Ollama for general AI responses
                                            threading.Thread(
                                                target=self.ask_llama, args=(command,), daemon=True
                                            ).start()
                                    else:
                                        # Bare wake word — acknowledge and enter conversational mode
                                        self._respond("Yes, I'm listening! What would you like me to do?")
                    except queue.Empty:
                        # Check for conversational timeout during idle periods
                        if self.conversational_mode:
                            if time.time() - self.last_activity_time > self.conversation_timeout:
                                self.conversational_mode = False
                                print("[VOSK] ✓ Conversational timeout - returning to wake word mode")
                                self._respond("I haven't heard from you for a while, so I'll go back to sleep. Say hey zyglass to wake me up.")
                        pass  # No audio in queue, continue listening
                    except Exception as e:
                        print(f"[VOSK] Recognition error: {e}")
        except Exception as e:
            print(f"[VOSK] Fatal audio error: {e}")
            self._enabled = False
    
    def _clear_wake(self) -> None:
        """Clear wake detection indicator for React UI."""
        with _voice_lock:
            _voice_state["wake_detected"] = False
            _voice_state["active"] = False
    
    def _route_fast(self, command: str) -> bool:
        """Fast local command routing — returns True if handled, False if should go to LLM."""
        global current_mode, currency_mode, lecture_mode
        
        words = command.lower().split()
        
        # Mode switching
        if any(w in words for w in ("switch", "change", "go", "mode")):
            if "object" in words or "detection" in words:
                with mode_lock:
                    current_mode = 1
                self._respond("Switching to object detection")
                return True
            elif "read" in words or "text" in words or "ocr" in words:
                with mode_lock:
                    current_mode = 2
                self._respond("Switching to text reader")
                return True
            elif "emotion" in words:
                with mode_lock:
                    current_mode = 3
                self._respond("Switching to emotion analyzer")
                return True
            elif "sign" in words or "gesture" in words:
                with mode_lock:
                    current_mode = 4
                self._respond("Switching to sign translator")
                return True
            elif "face" in words or "memory" in words:
                with mode_lock:
                    current_mode = 5
                self._respond("Switching to social memory")
                return True
        
        # Sub-mode toggles
        if "currency" in words or "money" in words or "rupee" in words:
            currency_mode = not currency_mode
            state = "enabled" if currency_mode else "disabled"
            self._respond(f"Currency reader {state}")
            return True
        
        if "lecture" in words and ("start" in words or "enable" in words):
            lecture_mode = True
            self._respond("Lecture mode enabled")
            return True
        
        if "lecture" in words and ("stop" in words or "disable" in words):
            lecture_mode = False
            self._respond("Lecture mode disabled")
            return True
        
        # Quick status queries
        if "what" in words and ("see" in words or "read" in words or "detect" in words):
            with ocr_lock:
                snap = list(ocr_results)
            if snap:
                combined = "; ".join(r["text"] for r in snap[:2])
                self._respond(f"I can see: {combined}")
            else:
                self._respond("No text detected right now")
            return True
        
        if "emotion" in words or "feeling" in words:
            self._respond(f"Current emotion: {current_emotion}")
            return True
        
        if "time" in words:
            current_time = time.strftime("%I:%M %p")
            self._respond(f"It's {current_time}")
            return True
        
        # Enhanced navigation commands
        if any(phrase in command.lower() for phrase in [
            "what can you do", "what are your features", "list features", "capabilities"
        ]):
            self._respond("I can detect objects, read text, recognize emotions, understand sign language, and identify faces. Say switch to plus mode name to change modes.")
            return True
        
        if any(phrase in command.lower() for phrase in [
            "what am i looking at", "what do you see", "describe what you see", "scan environment"
        ]):
            mode_name = modes_info.get(current_mode, {}).get("name", "unknown")
            if current_mode == 1:
                self._respond("I'm scanning for objects in your view")
            elif current_mode == 2:
                with ocr_lock:
                    snap = list(ocr_results)
                if snap:
                    text_content = "; ".join(r["text"] for r in snap[:3])
                    self._respond(f"I can read: {text_content}")
                else:
                    self._respond("I'm ready to read any text you point at")
            elif current_mode == 3:
                self._respond(f"I'm analyzing emotions. Current emotion: {current_emotion}")
            elif current_mode == 4:
                self._respond("I'm watching for sign language gestures")
            elif current_mode == 5:
                self._respond("I'm scanning for faces and people")
            else:
                self._respond(f"Currently in {mode_name} mode")
            return True
        
        # Simple responses that don't need Ollama
        if any(word in words for word in ["hello", "hi", "hey"]):
            self._respond("Hello! How can I help you?")
            return True
        
        if "help" in words:
            self._respond("I can detect objects, read text, recognize emotions, see hand signs, and identify faces")
            return True
        
        if any(word in words for word in ["thank", "thanks"]):
            self._respond("You're welcome!")
            return True
        
        if "status" in words:
            self._respond("All systems running normally")
            return True
        
        # Mode switching and navigation commands
        if any(word in words for word in ["switch", "change", "go to", "activate", "select"]):
            if "object" in words or "detection" in words or "first" in words or "one" in words:
                self._switch_mode(1)
                self._respond("Switching to object detection mode")
                return True
            elif "text" in words or "ocr" in words or "read" in words or "second" in words or "two" in words:
                self._switch_mode(2)
                self._respond("Switching to text reading mode")
                return True
            elif "emotion" in words or "feeling" in words or "third" in words or "three" in words:
                self._switch_mode(3)
                self._respond("Switching to emotion detection mode")
                return True
            elif "sign" in words or "gesture" in words or "fourth" in words or "four" in words:
                self._switch_mode(4)
                self._respond("Switching to sign language mode")
                return True
            elif "face" in words or "people" in words or "person" in words or "fifth" in words or "five" in words:
                self._switch_mode(5)
                self._respond("Switching to face recognition mode")
                return True

        # Mode switching and navigation commands
        if any(word in words for word in ["switch", "change", "go to", "activate", "select"]):
            if "object" in words or "detection" in words or "first" in words or "one" in words:
                self._switch_mode(1)
                self._respond("Switching to object detection mode")
                return True
            elif "text" in words or "ocr" in words or "read" in words or "second" in words or "two" in words:
                self._switch_mode(2)
                self._respond("Switching to text reading mode")
                return True
            elif "emotion" in words or "feeling" in words or "third" in words or "three" in words:
                self._switch_mode(3)
                self._respond("Switching to emotion detection mode")
                return True
            elif "sign" in words or "gesture" in words or "fourth" in words or "four" in words:
                self._switch_mode(4)
                self._respond("Switching to sign language mode")
                return True
            elif "face" in words or "people" in words or "person" in words or "fifth" in words or "five" in words:
                self._switch_mode(5)
                self._respond("Switching to face recognition mode")
                return True

        if "mode" in words:
            if "list" in words or "available" in words or "options" in words:
                self._respond("Available modes: object detection, text reading, emotion detection, sign language, and face recognition")
                return True
            else:
                mode_name = modes_info.get(current_mode, {}).get("name", str(current_mode))
                self._respond(f"Currently in {mode_name} mode")
                return True

        # Navigation and utility commands  
        if "next" in words:
            next_mode = (current_mode % 5) + 1
            self._switch_mode(next_mode)
            mode_name = modes_info.get(next_mode, {}).get("name", str(next_mode))
            self._respond(f"Switched to {mode_name}")
            return True

        if "previous" in words or "back" in words:
            prev_mode = ((current_mode - 2) % 5) + 1
            self._switch_mode(prev_mode)
            mode_name = modes_info.get(prev_mode, {}).get("name", str(prev_mode))
            self._respond(f"Switched to {mode_name}")
            return True

        # Navigation and utility commands  
        if "next" in words:
            next_mode = (current_mode % 5) + 1
            self._switch_mode(next_mode)
            mode_name = modes_info.get(next_mode, {}).get("name", str(next_mode))
            self._respond(f"Switched to {mode_name}")
            return True

        if "previous" in words or "back" in words:
            prev_mode = ((current_mode - 2) % 5) + 1
            self._switch_mode(prev_mode)
            mode_name = modes_info.get(prev_mode, {}).get("name", str(prev_mode))
            self._respond(f"Switched to {mode_name}")
            return True
        
        return False  # Not handled — send to LLM
    
    def _switch_mode(self, mode_id: int):
        """Switch to specified mode via API call."""
        try:
            import requests
            response = requests.post(f'http://localhost:5000/set_mode/{mode_id}', timeout=5)
            if response.status_code == 200:
                print(f"[VOICE] ✓ Mode switched to {mode_id}")
            else:
                print(f"[VOICE] ✗ Mode switch failed: {response.status_code}")
        except Exception as e:
            print(f"[VOICE] ✗ Mode switch error: {e}")

    def ask_llama(self, prompt: str):
        """Send command to Ollama for general AI responses."""
        if not OLLAMA_AVAILABLE:
            self._respond("AI assistant is offline")
            return
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"You are ZyGlass, an AI smart-glasses assistant. Give one short sentence (max 15 words). User says: {prompt}",
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)  # Increased timeout
            if response.status_code == 200:
                answer = response.json().get("response", "").strip()
                print(f"[LLAMA] Response: {answer}")
                self._respond(answer)
            else:
                print(f"[LLAMA] HTTP Error: {response.status_code}")
                self._respond("AI is processing, please try again")
        except requests.exceptions.Timeout:
            print("[LLAMA] Timeout error - Ollama is taking too long")
            self._respond("AI is thinking slowly, please wait and try again")
        except Exception as e:
            print(f"[LLAMA] Error: {e}")
            self._respond("AI assistant temporarily unavailable")
    
    def _respond(self, text: str) -> None:
        """Speak response and update shared state for React UI."""
        print(f"[VOSK] Response: '{text}'")
        self.audio_engine.speak(text)
        with _voice_lock:
            _voice_state["last_response"] = text
    
    def disable(self):
        """Disable the voice assistant."""
        self._enabled = False
        with _voice_lock:
            _voice_state["enabled"] = False
            _voice_state["listening"] = False


class VoiceAssistant:
    """
    Offline voice assistant for ZYGLASS.

    Architecture (two daemon threads, camera feed never affected):

      _listen_loop  ──────────────────────────────────────────────►
        Mic → SpeechRecognition VAD → WAV bytes → _audio_q (maxsize=1)

      _process_loop ──────────────────────────────────────────────►
        _audio_q → faster-whisper STT (RTX 4050 GPU, float16)
               → wake word filter ("hey zyglass" or "zyglass")
               → fast command router (mode switch, DB query, sub-mode toggle)
               → [not matched] → Ollama Llama 3 (existing connection)
               → pyttsx3 TTS (existing audio engine)

    Wake word detection uses the GPU Whisper transcription itself — no
    Porcupine license needed.  Background noise produces no transcription
    (Whisper outputs silence), so the wake filter acts as a first-pass VAD.

    Two-model strategy (reliability):
      • tiny.en  — loaded for wake detection on every phrase (12 ms on RTX 4050)
      • small.en — loaded once for high-accuracy command transcription
    On systems without CUDA both models fall back to CPU (int8 quantised).
    """

    # ── Mode name → mode number map for voice navigation ──────────────────
    _MODE_MAP: dict = {
        # Mode 1 keywords
        "insight":   1, "object":  1, "detect":  1, "detection": 1, "yolo": 1,
        "explore":   1, "explorer": 1,
        # Mode 2 keywords
        "read":      2, "reading":  2, "ocr":    2, "text":       2,
        "translate": 2, "multilingual": 2, "reader": 2,
        # Mode 3 keywords
        "emotion":   3, "feeling":  3, "mood":   3, "emotional":  3, "analyzer": 3,
        # Mode 4 keywords
        "sign":      4, "medical":  4, "hand":   4, "gesture":    4, "signing": 4,
        "translator": 4,
        # Mode 5 keywords
        "social":    5, "memory":   5, "face":   5, "recognize":  5, "recognition": 5,
        "who":       5,
    }

    def __init__(self, audio_engine: AudioAlertSystem) -> None:
        self.audio = audio_engine
        self._audio_q: queue.Queue = queue.Queue(maxsize=1)   # drop stale audio
        self._enabled = False
        self._wake_ts: float = 0.0   # time wake word last fired

        if not WHISPER_AVAILABLE or not SR_AVAILABLE:
            print("[VOICE] faster-whisper or SpeechRecognition missing — assistant disabled")
            return

        # ── Load Whisper models ────────────────────────────────────────────
        _dev, _ctype = ("cuda", "float16") if torch.cuda.is_available() else ("cpu", "int8")
        print(f"[VOICE] Loading Whisper models on {_dev} ({_ctype}) …")
        try:
            # tiny.en for fast wake detection (<15 ms on GPU)
            self._wake_model = WhisperModel("tiny.en",  device=_dev, compute_type=_ctype)
            # small.en for high-accuracy command transcription (~120 ms on GPU)
            self._cmd_model  = WhisperModel("small.en", device=_dev, compute_type=_ctype)
            print("[VOICE] Whisper tiny.en + small.en loaded ✓")
        except Exception as e:
            print(f"[VOICE] Whisper load failed: {e} — assistant disabled")
            return

        # ── SpeechRecognition microphone ───────────────────────────────────────
        try:
            self._recognizer = sr_lib.Recognizer()
            self._recognizer.pause_threshold   = 0.8   # give enough time to finish speaking
            self._recognizer.energy_threshold  = 200   # low baseline; we fix it below
            self._recognizer.dynamic_energy_threshold = False  # never drift upward and go deaf
            self._mic = sr_lib.Microphone(sample_rate=16000)
            # Calibrate once to set a stable threshold, then lock it
            with self._mic as src:
                self._recognizer.adjust_for_ambient_noise(src, duration=2.0)
            # Keep the calibrated value but clamp it — never above 400 so quiet speech is heard
            self._recognizer.energy_threshold = min(self._recognizer.energy_threshold, 400)
            print(f"[VOICE] Microphone calibrated ✓ energy_threshold={self._recognizer.energy_threshold:.0f}")
        except Exception as e:
            print(f"[VOICE] Microphone init failed: {e} — assistant disabled")
            return

        # ── Start threads ───────────────────────────────────────────────────
        self._enabled = True
        with _voice_lock:
            _voice_state["enabled"]   = True
            _voice_state["listening"] = True
        threading.Thread(target=self._listen_loop,  daemon=True, name="VA-Listen").start()
        threading.Thread(target=self._process_loop, daemon=True, name="VA-Process").start()
        print("[VOICE] Assistant ready — say 'Hey ZyGlass' to activate ✓")

    # ───────────────────────────────────────────────────────────────────
    # Thread 1 — Listen loop
    # ───────────────────────────────────────────────────────────────────
    def _listen_loop(self) -> None:
        """Continuously capture phrases from mic using SR energy-based VAD.
        Each captured phrase is WAV-encoded and pushed to _audio_q.
        maxsize=1 means: if _process_loop is still busy, the stale phrase
        is dropped and the fresh one takes its place (always-latest policy).
        """
        with self._mic as src:
            while True:
                try:
                    with _voice_lock:
                        _voice_state["listening"] = True
                    print("[VOICE] Listening for speech...")
                    # phrase_time_limit=5: cap utterance at 5 s (handles fast commands)
                    audio_data = self._recognizer.listen(src, phrase_time_limit=5)
                    print("[VOICE] Audio captured, processing...")
                    with _voice_lock:
                        _voice_state["listening"] = False
                        _voice_state["active"]    = True
                    wav_bytes = audio_data.get_wav_data()
                    try:
                        self._audio_q.get_nowait()   # drop stale frame
                    except queue.Empty:
                        pass
                    self._audio_q.put_nowait(wav_bytes)
                except Exception as e:
                    print(f"[VOICE] Listen error: {e}")
                    time.sleep(0.5)   # mic error — brief back-off before retry

    # ───────────────────────────────────────────────────────────────────
    # Thread 2 — Process loop
    # ───────────────────────────────────────────────────────────────────
    def _process_loop(self) -> None:
        """Pop WAV from queue, run two-stage Whisper pipeline, route command."""
        while True:
            try:
                wav_bytes = self._audio_q.get(timeout=2.0)
            except queue.Empty:
                with _voice_lock:
                    _voice_state["active"] = False
                continue

            # Honor /toggle_voice — discard queued audio when disabled
            with _voice_lock:
                if not _voice_state.get("enabled", True):
                    _voice_state["active"] = False
                    continue

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                tmp.write(wav_bytes)
                tmp.flush()
                tmp_path = tmp.name
            finally:
                tmp.close()

            try:
                # ── Stage 1: fast tiny.en wake detection ───────────────────────
                segs, _ = self._wake_model.transcribe(
                    tmp_path, beam_size=1, language="en",
                    vad_filter=True,          # skip silent segments entirely
                    vad_parameters={"min_silence_duration_ms": 300},
                )
                tiny_text = " ".join(s.text for s in segs).strip().lower()
                print(f"[VOICE] Transcribed (tiny): '{tiny_text}'")

                # ── Wake word gate ────────────────────────────────────────────
                # Whisper transcribes the invented word "ZyGlass" phonetically in
                # many different ways. We cast a wide net over all plausible outputs.
                _WAKE_VARIANTS = (
                    "zyglass",   "zy glass",  "zig glass", "ziglass",
                    "zig lass",  "z glass",   "siglass",   "si glass",
                    "sea glass", "syglass",   "sy glass",  "c glass",
                    "hi glass",  "aiglass",   "ai glass",  "eye glass",
                    "eyeglass",  "i glass",   "zig-glass", "ziglas",
                    "z-glass",   "glass",     # fallback — short utterances of just "glass"
                )
                if not any(v in tiny_text for v in _WAKE_VARIANTS):
                    print("[VOICE] No wake word detected, ignoring")
                    continue   # background chatter — discard

                print("[VOICE] ✓ WAKE WORD DETECTED!")

                # Mark wake in shared state (React UI flashes indicator for 2 s)
                with _voice_lock:
                    _voice_state["wake_detected"] = True
                self._wake_ts = time.time()
                threading.Timer(2.0, self._clear_wake).start()

                # ── Stage 2: high-accuracy small.en command transcription ─────
                segs2, _ = self._cmd_model.transcribe(
                    tmp_path, beam_size=5, language="en",
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 300},
                )
                full_text = " ".join(s.text for s in segs2).strip()
                # Strip the wake word itself to isolate the command
                command = full_text.lower()
                for ww in (
                    "hey zyglass", "zyglass", "zy glass", "zig glass", "ziglass",
                    "zig lass", "z glass", "siglass", "si glass", "sea glass",
                    "syglass", "sy glass", "c glass", "hi glass", "aiglass",
                    "ai glass", "eye glass", "eyeglass", "i glass", "zig-glass",
                    "z-glass", "hey glass",
                ):
                    command = command.replace(ww, "").strip()
                command = command.strip(" ,.").strip()

                print(f"[VOICE] Command: '{command}'")
                with _voice_lock:
                    _voice_state["last_command"] = command or "(wake word only)"
                    _voice_state["last_ts"]      = time.strftime("%H:%M:%S")

                if command:
                    self._route(command)
                else:
                    # Bare wake word — just acknowledge so the user knows it's listening
                    self._respond("Yes? Go ahead.")

            except Exception as e:
                print(f"[VOICE] Transcription error: {e}")
            finally:
                with _voice_lock:
                    _voice_state["active"] = False
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _clear_wake(self) -> None:
        with _voice_lock:
            _voice_state["wake_detected"] = False

    # ───────────────────────────────────────────────────────────────────
    # Command Router
    # ───────────────────────────────────────────────────────────────────
    def _route(self, command: str) -> None:
        """Rule-based fast path first, Ollama fallback for everything else.
        Fast path has <5 ms latency; Ollama path 2-10 s but still async.
        """
        global current_mode, currency_mode, lecture_mode

        words = command.lower().split()

        # ── 1. Mode switching ────────────────────────────────────────────────
        if any(w in words for w in ("switch", "change", "go", "activate", "mode", "enable")):
            for word in words:
                if word in self._MODE_MAP:
                    target = self._MODE_MAP[word]
                    with mode_lock:
                        current_mode = target
                    mode_name = modes_info.get(target, {}).get("name", str(target))
                    response = f"Switching to {mode_name}"
                    self._respond(response)
                    return

        # ── 2. Sub-mode toggles ───────────────────────────────────────────────
        if any(w in words for w in ("currency", "money", "rupee", "cash", "note", "banknote")):
            currency_mode = not currency_mode
            state = "enabled" if currency_mode else "disabled"
            self._respond(f"Currency reader {state}")
            return

        if any(w in words for w in ("lecture", "notes", "whiteboard", "class", "classroom")):
            lecture_mode = not lecture_mode
            state = "enabled" if lecture_mode else "disabled"
            self._respond(f"Lecture logger {state}")
            return

        # ── 3. DB queries — answered directly from MongoDB / local state ────────
        if any(w in words for w in ("who", "visited", "seen", "recognized")):
            if MONGO_AVAILABLE and db_faces is not None:
                try:
                    today = time.strftime("%Y-%m-%dT")
                    docs  = list(db_faces.find(
                        {"timestamp": {"$gte": today}}, {"_id": 0}
                    ).sort("unix_ts", -1).limit(10))
                    names = list(dict.fromkeys(d["name"] for d in docs))
                    if names:
                        self._respond("Today you met: " + ", ".join(names))
                    else:
                        self._respond("No one has been recognised today yet.")
                except Exception:
                    self._respond("Could not reach the database right now.")
            else:
                # Fallback to in-memory face history
                names = list(dict.fromkeys(
                    e["name"] for e in face_history if e["name"] != "Unknown"
                ))
                if names:
                    self._respond("Recognised faces this session: " + ", ".join(names))
                else:
                    self._respond("No faces recognised in this session yet.")
            return

        if any(w in words for w in ("sign", "signs", "gesture", "gestures")):
            last = sign_history[-1]["sign"] if sign_history else "none"
            self._respond(f"Last sign detected: {last}")
            return

        if any(w in words for w in ("denomination", "rupees", "value", "amount")):
            last_d = mode_stats[1].get("last_denomination", "none")
            self._respond(f"Last detected note: {last_d}")
            return

        if any(w in words for w in ("emotion", "feeling", "mood")):
            self._respond(f"Current emotion detected: {current_emotion}")
            return

        if any(w in words for w in ("summary", "summarize", "notes", "lecture")):
            summary = _lecture_summary[0]
            if summary:
                first = summary.split("\n")[0][:120]
                self._respond(f"Lecture summary: {first}")
            else:
                self._respond("No lecture notes captured yet. Enable lecture mode first.")
            return

        if any(w in words for w in ("what", "read")):
            with ocr_lock:
                snap = list(ocr_results)
            if snap:
                combined = "; ".join(r["text"] for r in snap[:3])
                self._respond(f"I can see: {combined}")
            else:
                self._respond("No text detected right now.")
            return

        if any(w in words for w in ("status", "mode", "current")):
            name = modes_info.get(current_mode, {}).get("name", str(current_mode))
            self._respond(f"Currently in {name} mode.")
            return

        # ── 4. Help command ───────────────────────────────────────────────────
        if any(w in words for w in ("help", "commands", "what can you do")):
            self._respond(
                "I can switch modes, toggle currency reader, toggle lecture logger, "
                "tell you who visited, read detected text, check your emotion, "
                "and answer general questions. Say Hey ZyGlass followed by your command."
            )
            return

        # ── 5. Ollama fallback for everything else ──────────────────────────
        if OLLAMA_AVAILABLE:
            threading.Thread(
                target=self._ask_ollama, args=(command,), daemon=True
            ).start()
        else:
            self._respond("I didn't understand that. Say 'help' for a list of commands.")

    def _ask_ollama(self, command: str) -> None:
        """Send command to the existing Ollama connection. Runs in own thread."""
        mode_name = modes_info.get(current_mode, {}).get("name", "")
        prompt = (
            f"You are ZyGlass, an AI smart-glasses assistant worn by a visually impaired user. "
            f"The glasses are currently in '{mode_name}' mode. "
            f"Give a single, short, spoken-English sentence response (max 20 words). "
            f"No markdown, no lists. User said: {command}"
        )
        try:
            resp = _ollama_lib.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 60},
            )
            answer = resp["message"]["content"].strip()
            self._respond(answer)
        except Exception as e:
            print(f"[VOICE] Ollama error: {e}")
            self._respond("I'm having trouble connecting to the language model.")

    def _respond(self, text: str) -> None:
        """Speak response and update shared state for React UI."""
        print(f"[VOICE] Response: '{text}'")
        self.audio.speak(text)
        with _voice_lock:
            _voice_state["last_response"] = text


# Instantiate voice assistant — starts its own daemon threads
if VOSK_AVAILABLE:
    voice_assistant = FastVoiceAssistant(audio)
else:
    # Fallback to Whisper-based assistant if Vosk not available
    try:
        voice_assistant = VoiceAssistant(audio)
    except Exception as e:
        print(f"[VOICE] Both Vosk and Whisper assistants failed: {e}")
        voice_assistant = None
# (last_seen_objects, last_ocr_text, last_emotion, last_sign, sign_cooldown, last_seen_faces)

# Mode 4 — sign (3 s cooldown) — also used by set_mode() for reset
last_spoken_sign: list = [None]
last_spoken_time: list = [0.0]
TTS_COOLDOWN = 3.0

# ── MongoDB logging helpers ───────────────────────────────────────────────────
def _mongo_log_face(name: str) -> None:
    """Insert a face recognition event into MongoDB (fire-and-forget)."""
    if not MONGO_AVAILABLE or db_faces is None:
        return
    try:
        db_faces.insert_one({
            "name":      name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "unix_ts":   time.time(),
        })
    except Exception as e:
        print(f"[MONGO] face log error: {e}")


def _mongo_log_sign(sign: str) -> None:
    """Insert a confirmed medical sign into MongoDB (fire-and-forget)."""
    if not MONGO_AVAILABLE or db_signs is None:
        return
    try:
        db_signs.insert_one({
            "sign":      sign,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "unix_ts":   time.time(),
        })
    except Exception as e:
        print(f"[MONGO] sign log error: {e}")


def _mongo_log_ocr(raw_text: str, simplified: str) -> None:
    """Insert an OCR reading + its LLM simplification into MongoDB."""
    if not MONGO_AVAILABLE or db_ocr is None:
        return
    try:
        db_ocr.insert_one({
            "raw_text":   raw_text,
            "simplified": simplified,
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "unix_ts":    time.time(),
        })
    except Exception as e:
        print(f"[MONGO] OCR log error: {e}")


# ── Ollama LLM helper — simplifies medical prescription text ──────────────────
_llm_cache: dict = {}    # raw_text → simplified text (session cache, avoids re-querying)
_llm_lock = threading.Lock()
_llm_last_result: list = [""]    # [simplified_text] updated by async LLM background thread
_llm_pending:     list = [False] # [True] while an LLM request is in-flight (throttle)

# Background I/O pool — replaces per-detection threading.Thread(...).start().
# Reuses 4 daemon threads instead of creating+destroying one per event.
_bg_pool = _cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="bg_io")

# Translation cache — OCR text repeats every 1 s; caching avoids repeat HTTP round-trips.
_translation_cache: dict = {}  # text → translated string, capped at 1000 entries


def _llm_simplify(raw_text: str) -> str:
    """Ask Llama 3.2 to turn prescription/medical text into plain English.

    Returns simplified string on success, or raw_text on failure/timeout.
    Results are cached in _llm_cache to avoid re-querying identical text.
    Runs synchronously — call only from the ocr_worker background thread.
    """
    if not OLLAMA_AVAILABLE or not raw_text.strip():
        return raw_text
    with _llm_lock:
        if raw_text in _llm_cache:
            return _llm_cache[raw_text]
    prompt = (
        "You are a helpful medical assistant for elderly patients. "
        "Rewrite the following prescription label text in simple plain English "
        "that a patient with no medical training can understand. "
        "Keep it to 1-2 short sentences. Do not add extra commentary.\n\n"
        f"Label text: {raw_text}\n\nSimple explanation:"
    )
    try:
        response = _ollama_lib.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 80},
        )
        simplified = response["message"]["content"].strip()
        with _llm_lock:
            _llm_cache[raw_text] = simplified
        print(f"[LLM] '{raw_text[:40]}...' → '{simplified[:60]}...'")
        return simplified
    except Exception as e:
        print(f"[LLM] Simplification error: {e}")
        return raw_text

def _llm_simplify_async(raw_text: str) -> None:
    """Fire-and-forget wrapper — runs _llm_simplify in a background thread.
    ocr_worker never blocks waiting for Llama 3.2 inference (2-10 s).
    Result is stored in _llm_last_result[0] for the next OCR draw cycle.
    """
    try:
        result = _llm_simplify(raw_text)
        _llm_last_result[0] = result
    except Exception:
        pass
    finally:
        _llm_pending[0] = False

# Mode 5 — face (10 s cooldown) — also used by set_mode() for reset
last_spoken_face: list = [None]
last_spoken_face_time: list = [0.0]
FACE_TTS_COOLDOWN = 10.0

emotion_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

url = "http://10.56.198.212:81/stream"  # Your ESP32-CAM working address
vs = VideoStream(url).start()

def get_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calc_ear(eye):
    """Calculate Eye Aspect Ratio (EAR)."""
    A = get_distance(eye[1], eye[5])
    B = get_distance(eye[2], eye[4])
    C = get_distance(eye[0], eye[3])
    bottom = np.multiply(2.0, C)
    return (A + B) / bottom

def calc_mar(points):
    """Calculate Mouth Aspect Ratio (MAR)."""
    v_dist = get_distance(points[13], points[14])
    h_dist = get_distance(points[78], points[308])
    return v_dist / h_dist

def yolo_worker() -> None:
    """
    YOLO inference in a dedicated thread — AI thread only draws cached boxes.
    Running YOLO inline blocked the display pipeline for 80-150 ms per frame;
    here the AI thread feeds the latest raw frame and moves on immediately.
    """
    print("[YOLO THREAD] YOLO worker started")
    last_yolo_ts = 0.0
    while pipeline_running:
        with mode_lock:
            is_mode1 = (current_mode == 1)
        if not is_mode1:
            time.sleep(0.05)
            continue
        with _yolo_src_lock:
            src    = _yolo_src_frame[0]
            src_ts = _yolo_src_frame[1]
        if src is None or src_ts <= last_yolo_ts:
            time.sleep(0.01)   # no new frame yet
            continue
        last_yolo_ts = src_ts
        try:
            boxes_out: list = []
            results = model(src, stream=True, conf=0.3, imgsz=320, verbose=False)
            for r in results:
                total_kept = 0
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # ── Size filter: ignore tiny blobs (noise / distant objects) ──
                    area = (x2 - x1) * (y2 - y1)
                    if area < 200:  # Reduced from 400 for QVGA resolution
                        continue
                    label = model.names[int(box.cls[0])]
                    conf  = float(box.conf[0])
                    boxes_out.append({"label": f"{label} {conf:.0%}",
                                      "raw":   label,
                                      "tl":    (x1, y1),
                                      "br":    (x2, y2)})
                    total_kept += 1
                mode_stats[1]["objects_detected"] = total_kept
            with _yolo_boxes_lock:
                _yolo_boxes[:] = boxes_out
        except Exception as e:
            print(f"[YOLO THREAD] Error: {e}")
        # 0.25 s sleep caps to ~4 Hz — reduced from 7 Hz for CPU stability

        time.sleep(0.25)


def currency_worker() -> None:
    """
    Currency Reader — dedicated YOLO thread for Indian Rupee banknote detection.
    Runs as a sub-mode of Mode 1: activates only when currency_mode=True.
    Inference uses the lighter rupee_model (≤320 px) at ~5 Hz to keep CPU free.

    Pipeline:
      process_ai_thread → _currency_src_frame → currency_worker → _currency_boxes
      generate_frames reads _currency_boxes and draws denomination banners.

    TTS: speaks the dominant denomination once per CURRENCY_TTS_COOLDOWN seconds
    so visually impaired users hear "Ten Rupees" / "Five Hundred Rupees" etc.
    """
    print("[CURRENCY THREAD] Currency Reader worker started")
    # Map YOLO class names → clean spoken forms
    _DENOM_SPEECH = {
        "10": "Ten Rupees",    "20": "Twenty Rupees",  "50": "Fifty Rupees",
        "100": "Hundred Rupees", "200": "Two Hundred Rupees",
        "500": "Five Hundred Rupees", "2000": "Two Thousand Rupees",
    }
    last_currency_ts = 0.0
    while pipeline_running:
        with mode_lock:
            is_mode1 = (current_mode == 1)
        if not is_mode1 or not currency_mode or not RUPEE_MODEL_AVAILABLE:
            time.sleep(0.1)
            continue
        with _currency_src_lock:
            src    = _currency_src_frame[0]
            src_ts = _currency_src_frame[1]
        if src is None or src_ts <= last_currency_ts:
            time.sleep(0.01)
            continue
        last_currency_ts = src_ts
        try:
            boxes_out: list = []
            results = rupee_model(src, stream=True, conf=0.45, imgsz=320, verbose=False)  # type: ignore[misc]
            best_denom = ""
            best_conf  = 0.0
            for r in results:
                for box in r.boxes:
                    # cspell:ignore xyxy
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area < 500:
                        continue
                    raw_name = rupee_model.names[int(box.cls[0])]  # type: ignore[union-attr]   # e.g. "500"
                    conf     = float(box.conf[0])
                    spoken   = _DENOM_SPEECH.get(str(raw_name), f"{raw_name} Rupees")
                    boxes_out.append({
                        "denomination": spoken,
                        "raw":          str(raw_name),
                        "conf":         conf,
                        "tl":           (x1, y1),
                        "br":           (x2, y2),
                    })
                    if conf > best_conf:
                        best_conf  = conf
                        best_denom = spoken
                mode_stats[1]["notes_detected"] = len(boxes_out)
                if boxes_out:
                    mode_stats[1]["last_denomination"] = best_denom
            with _currency_boxes_lock:
                _currency_boxes[:] = boxes_out
            # TTS — speak dominant denomination, debounced
            now = time.time()
            if (best_denom
                    and (best_denom != _currency_last_spoken[0]
                         or now - _currency_last_time[0] > CURRENCY_TTS_COOLDOWN)):
                audio.speak(best_denom)
                _currency_last_spoken[0] = best_denom
                _currency_last_time[0]   = now
        except Exception as e:
            print(f"[CURRENCY THREAD] Error: {e}")
        time.sleep(0.2)    # ~5 Hz — improved currency detection rate


def lecture_worker() -> None:
    """
    Lecture Logger — whiteboard-optimised OCR sub-mode of Mode 2.
    Activates when lecture_mode=True.  Uses heavier preprocessing tuned for
    dark-marker text on a white/green board seen through an ESP32-CAM:
      • Convert to greyscale + adaptive threshold → eliminates board glare
      • Full-frame pass (not centre-crop) to catch full-width equations
      • Stability filter: text must survive 2 consecutive passes
      • Every LECTURE_SUMMARIZE_EVERY seconds, fires async Ollama to produce
        a concise bullet-point summary streamed to the React UI via /lecture_data

    Notes accumulate in _lecture_lines (max 200 entries, rolling).
    """
    print("[LECTURE THREAD] Lecture Logger worker started")
    _hist_texts: list = []
    _hist_data:  list = []
    STABLE_WIN   = 2
    CONF_THRESH  = 0.55
    last_lecture_ts = 0.0

    def _summarize_async(raw_text: str) -> None:
        """Fire-and-forget Ollama summarization into _lecture_summary."""
        if not OLLAMA_AVAILABLE or not raw_text.strip():
            _lecture_pending[0] = False
            return
        prompt = (
            "You are a precise academic note-taker. "
            "The following text was captured from a university whiteboard during an AI lecture. "
            "Produce concise bullet-point notes (max 8 bullets, plain English). "
            "Preserve any formulas exactly as written. "
            "Do not add commentary not present in the text.\n\n"
            f"Whiteboard text:\n{raw_text}\n\nBullet-point notes:"
        )
        try:
            resp = _ollama_lib.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 200},
            )
            summary = resp["message"]["content"].strip()
            _lecture_summary[0] = summary
            mode_stats[2]["last_summary"] = summary[:120]
            print(f"[LECTURE] Summary updated ({len(summary)} chars)")
        except Exception as e:
            print(f"[LECTURE] LLM error: {e}")
        finally:
            _lecture_pending[0] = False

    while pipeline_running:
        with mode_lock:
            is_mode2 = (current_mode == 2)
        if not is_mode2 or not lecture_mode:
            time.sleep(0.5)
            continue
        with _lecture_src_lock:
            src    = _lecture_src_frame[0]
            src_ts = _lecture_src_frame[1]
        if src is None or src_ts <= last_lecture_ts:
            time.sleep(0.1)
            continue
        last_lecture_ts = src_ts
        try:
            # ── Whiteboard preprocessing ──────────────────────────────────────
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # Adaptive threshold: handles uneven board lighting & ESP32 glare
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                blockSize=31, C=12
            )
            # Unsharp mask on the binary — sharpens letter edges before OCR
            blurred = cv2.GaussianBlur(thresh, (0, 0), sigmaX=1.5)
            sharp   = cv2.addWeighted(thresh, 1.8, blurred, -0.8, 0)

            # Full-frame OCR — whiteboard text spans the whole width
            raw, _ = ocr_engine(sharp)  # type: ignore[misc]

            run_texts: set  = set()
            run_data:  list = []
            if raw:
                for line in raw:
                    box  = line[0]
                    text = str(line[1]).strip()
                    conf = float(line[2])
                    if conf < CONF_THRESH or len(text) < 3:
                        continue
                    tl = [int(box[0][0]), int(box[0][1])]
                    br = [int(box[2][0]), int(box[2][1])]
                    run_texts.add(text)
                    run_data.append({"text": text, "conf": int(conf * 100), "tl": tl, "br": br})

            # Stability filter
            _hist_texts.append(run_texts)
            _hist_data.append(run_data)
            if len(_hist_texts) > STABLE_WIN:
                _hist_texts.pop(0); _hist_data.pop(0)
            if len(_hist_texts) >= STABLE_WIN:
                stable_set = _hist_texts[0].intersection(*_hist_texts[1:])
                stable = [r for r in run_data if r["text"] in stable_set]
            else:
                stable = run_data

            if stable:
                ts = time.strftime("%H:%M:%S")
                with _lecture_lock:
                    for r in stable:
                        entry = {"text": r["text"], "time": ts,
                                 "conf": r["conf"], "tl": r["tl"], "br": r["br"]}
                        # De-duplicate vs last 5 entries
                        recent_texts = {e["text"] for e in _lecture_lines[-5:]}
                        if r["text"] not in recent_texts:
                            _lecture_lines.append(entry)
                    if len(_lecture_lines) > 200:
                        _lecture_lines[:] = _lecture_lines[-200:]
                mode_stats[2]["lecture_lines"] = len(_lecture_lines)

                # Periodic AI summarization
                now = time.time()
                if (not _lecture_pending[0]
                        and now - _lecture_last_summary_ts[0] > LECTURE_SUMMARIZE_EVERY):
                    combined = "\n".join(r["text"] for r in _lecture_lines[-40:])
                    if combined.strip() != _lecture_last_raw[0]:
                        _lecture_pending[0]       = True
                        _lecture_last_raw[0]      = combined
                        _lecture_last_summary_ts[0] = now
                        _bg_pool.submit(_summarize_async, combined)
        except Exception as e:
            print(f"[LECTURE THREAD] Error: {e}")
        time.sleep(LECTURE_OCR_INTERVAL)


def emotion_worker() -> None:
    """
    Haar cascade + ResNet50 emotion inference in a dedicated thread.
    Replaces the inline Mode 3 block that could block up to 80 ms per frame.
    """
    print("[EMOTION THREAD] Emotion worker started")
    ROLLING_WINDOW = 5   # store last N predictions, output the most frequent
    last_emo_ts = 0.0
    frame_counter = 0  # Skip frames for performance
    while pipeline_running:
        with mode_lock:
            is_mode3 = (current_mode == 3)
        if not is_mode3:
            time.sleep(0.05)
            continue
        with _emotion_src_lock:
            src    = _emotion_src_frame[0]
            src_ts = _emotion_src_frame[1]
        if src is None or src_ts <= last_emo_ts:
            time.sleep(0.01)
            continue
        
        # Skip every other frame to reduce CPU load
        frame_counter += 1
        if frame_counter % 2 == 0:
            last_emo_ts = src_ts
            time.sleep(0.05)
            continue
            
        last_emo_ts = src_ts
        try:
            gray      = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # Reduce to 1/4 resolution for even faster Haar cascade
            gray_quarter = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
            raw_dets  = face_cascade.detectMultiScale(
                gray_quarter, scaleFactor=1.3, minNeighbors=3,
                minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
            # Scale bbox coordinates back to full-resolution frame
            faces = [(x*4, y*4, w*4, h*4) for (x, y, w, h) in raw_dets] if len(raw_dets) else []
            boxes_out: list = []
            emotion_out = "Detecting"
            for (x, y, w, h) in faces:
                if (y > 10 and x > 10
                        and y + h < src.shape[0] - 10
                        and x + w < src.shape[1] - 10
                        and w > 40 and h > 40):
                    try:
                        roi = src[y:y+h, x:x+w]
                        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))  # type: ignore
                        input_tensor = emotion_transform(roi_pil).unsqueeze(0).to(device)  # type: ignore[union-attr]
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(  # type: ignore[attr-defined]
                                    enabled=torch.cuda.is_available()):
                                raw_out = emotion_model(input_tensor)  # type: ignore[operator]
                                outputs = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
                            probs = torch.nn.functional.softmax(outputs.float(), dim=1)[0]
                            confidence, preds = torch.max(probs, 0)
                            if confidence.item() * 100 > 30:
                                raw_emotion = emotion_labels[int(preds.item())]
                                # ── Rolling majority vote (5-frame anti-flicker window) ──
                                _emotion_history.append(raw_emotion)
                                if len(_emotion_history) > ROLLING_WINDOW:
                                    _emotion_history.pop(0)
                                counts_e = Counter(_emotion_history)
                                emotion_out = counts_e.most_common(1)[0][0]
                    except Exception:
                        pass
                    boxes_out.append((x, y, w, h))
                    break   # only first valid face per pass for speed
            with _emotion_cache_lock:
                _emotion_cache["boxes"]   = boxes_out
                _emotion_cache["emotion"] = emotion_out
        except Exception as e:
            print(f"[EMOTION THREAD] Error: {e}")
        time.sleep(0.5)    # ~2 Hz emotion detection — aggressive reduction for WiFi


def _align_face(img_rgb: np.ndarray, top: int, right: int, bottom: int, left: int) -> np.ndarray:
    """
    Geometric face alignment: rotate the crop so both eyes are perfectly horizontal.
    Uses dlib eye landmarks returned by face_recognition for the rotation angle.
    Falls back to unrotated crop if landmarks are unavailable.
    """
    try:
        lms = face_recognition.face_landmarks(  # type: ignore
            img_rgb, [(top, right, bottom, left)])
        if not lms:
            return img_rgb[top:bottom, left:right]
        left_eye  = np.mean(lms[0]["left_eye"],  axis=0)
        right_eye = np.mean(lms[0]["right_eye"], axis=0)
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle) < 1.0:          # already horizontal enough
            return img_rgb[top:bottom, left:right]
        cy = (top + bottom) // 2
        cx = (left + right) // 2
        h_crop = bottom - top
        w_crop = right - left
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(img_rgb, M, (img_rgb.shape[1], img_rgb.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return rotated[max(cy - h_crop // 2, 0): cy + h_crop // 2,
                       max(cx - w_crop // 2, 0): cx + w_crop // 2]
    except Exception:
        return img_rgb[top:bottom, left:right]


def face_worker() -> None:
    """
    HOG face_recognition scan in a dedicated thread — AI thread draws cached boxes.
    Replaces the every-6-frames inline block that blocked for 100-300 ms.
    """
    print("[FACE THREAD] Social Memory worker started")
    last_face_ts = 0.0
    while pipeline_running:
        with mode_lock:
            is_mode5 = (current_mode == 5)
        if not is_mode5 or not FACE_REC_AVAILABLE:
            time.sleep(0.1)
            continue
        with _face_worker_lock:
            src    = _face_worker_frame[0]
            src_ts = _face_worker_frame[1]
        if src is None or src_ts <= last_face_ts:
            time.sleep(0.05)
            continue
        last_face_ts = src_ts
        try:
            # Reduce to 1/6 scale for much faster face recognition (was 1/4)
            small     = cv2.resize(src, (0, 0), fx=0.16, fy=0.16)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb_small, model="hog")  # type: ignore
            # Single-pass encoding — removed the expensive _align_face re-encode path
            # (it ran face_locations + face_encodings a second time per face: 2× cost)
            encs = face_recognition.face_encodings(rgb_small, locs)  # type: ignore
            names_out: list = []
            for enc in encs:
                name = "Unknown"
                if known_face_encodings:
                    matches = face_recognition.compare_faces(  # type: ignore
                        known_face_encodings, enc, tolerance=0.5)
                    if True in matches:
                        dists = face_recognition.face_distance(known_face_encodings, enc)  # type: ignore
                        best  = int(np.argmin(dists))
                        if matches[best]:
                            name = known_face_names[best]
                names_out.append(name)
            # Commit results atomically via slice assignment (no separate lock needed)
            _face_locations[:] = locs
            _face_names[:]     = names_out
            # Update rolling history
            ts_str = time.strftime("%H:%M:%S")
            for n in names_out:
                face_history.append({"name": n, "time": ts_str})
                # Log known faces to MongoDB via pool (no Unknown spam)
                if n != "Unknown":
                    _bg_pool.submit(_mongo_log_face, n)
            if len(face_history) > 20:
                face_history[:] = face_history[-20:]
            mode_stats[5]["faces_recognized"] = len([n for n in names_out if n != "Unknown"])
            mode_stats[5]["known_faces"] = list(
                dict.fromkeys(n for n in names_out if n != "Unknown"))
        except Exception as e:
            print(f"[FACE THREAD] Error: {e}")
        time.sleep(FACE_SCAN_INTERVAL)


def sign_worker() -> None:
    """
    Mode 4: MediaPipe hand-landmark + ASL model inference in a dedicated thread.
    Moves all blocking MediaPipe/sklearn calls out of process_ai_thread so the
    live feed stays at 30 fps regardless of inference speed (~50 ms per call).
    """
    print("[SIGN THREAD] Sign worker started")
    _local_last_sign:     str   = ""
    _local_sign_cooldown: float = 0.0
    last_sign_ts = 0.0
    while pipeline_running:
        with mode_lock:
            is_mode4 = (current_mode == 4)
        if not is_mode4:
            time.sleep(0.1)
            continue
        with _sign_src_lock:
            src    = _sign_src_frame[0]
            src_ts = _sign_src_frame[1]
        if src is None or src_ts <= last_sign_ts:
            time.sleep(0.01)
            continue
        last_sign_ts = src_ts
        try:
            if not asl_model_available or mp_hands_detector is None:
                with _sign_cache_lock:
                    _sign_cache["sign"]      = "Model Not Loaded"
                    _sign_cache["landmarks"] = None
                entry = {"sign": "Model Not Loaded", "time": time.strftime("%H:%M:%S")}
                sign_history.append(entry)
                if len(sign_history) > 10:
                    sign_history.pop(0)
                time.sleep(0.5)
                continue
            frame_rgb    = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            hand_results = mp_hands_detector.detect(mp_image)
            raw_sign      = "No Hand Detected"
            landmarks_out = None
            if hand_results.hand_landmarks:
                landmarks     = hand_results.hand_landmarks[0]
                landmarks_out = landmarks
                norm_row = normalize_landmarks(landmarks)
                if len(norm_row) == 63 and asl_model is not None:
                    raw_sign = str(asl_model.predict([norm_row])[0])  # type: ignore[union-attr]
            # ── Temporal majority vote ─────────────────────────────────────
            if raw_sign == "No Hand Detected":
                _sign_votes.clear()
                detected_sign = "No Hand Detected"
            else:
                _sign_votes.append(raw_sign)
                if len(_sign_votes) > VOTE_WINDOW:
                    _sign_votes.pop(0)
                counts = Counter(_sign_votes)
                top_sign, top_count = counts.most_common(1)[0]
                detected_sign = top_sign if top_count >= VOTE_THRESHOLD else raw_sign
                mode_stats[4]["signs_detected"] += 1
                mode_stats[4]["last_sign"] = detected_sign
            # Update shared cache — AI thread reads this to draw HUD each frame
            with _sign_cache_lock:
                _sign_cache["sign"]      = detected_sign
                _sign_cache["landmarks"] = landmarks_out
            # Append to rolling history (max 10 entries)
            entry = {"sign": detected_sign, "time": time.strftime("%H:%M:%S")}
            sign_history.append(entry)
            if len(sign_history) > 10:
                sign_history.pop(0)
            # TTS — debounced, 3 s cooldown per unique sign
            current_time = time.time()
            if (detected_sign not in ("No Hand Detected", "Model Not Loaded")
                    and (detected_sign != _local_last_sign
                         or (current_time - _local_sign_cooldown) > 3)):
                audio.speak(detected_sign)
                _local_last_sign     = detected_sign
                _local_sign_cooldown = current_time
                _bg_pool.submit(_mongo_log_sign, detected_sign)
        except Exception as e:
            print(f"[SIGN THREAD] Error: {e}")
        time.sleep(0.4)    # ~2.5 Hz hand gesture — aggressive reduction for WiFi


def capture_frames_thread():
    """
    THREAD 1: Frame Capture
    Pulls frames from VideoStream into capture_queue only when a NEW JPEG has
    arrived (tracked via vs.timestamp).  Deduplication means the AI thread never
    burns time re-processing the same image, which was a major stutter source.
    """
    print("[THREAD 1] Frame Capture thread started")
    last_ts = 0.0
    while pipeline_running:
        # Skip if camera not connected
        if vs is None:
            time.sleep(0.1)
            continue
        # Read timestamp + frame under the same lock to stay consistent
        with vs.lock:
            ts  = vs.timestamp
            raw = vs.raw_jpeg          # bytes — immutable, safe outside lock
        if raw is not None and ts > last_ts and (time.time() - ts) < FRAME_TIMEOUT:
            arr   = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            last_ts = ts
        else:
            frame = None

        if frame is None:
            time.sleep(0.001)   # minimal yield for faster frame processing
            continue

        # Evict stale frame, push fresh one
        try:
            capture_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            capture_queue.put_nowait(frame)
        except queue.Full:
            pass

def ocr_worker() -> None:
    """
    THREAD 5 (Mode 2): Dedicated OCR Worker — PaddleOCR via RapidOCR ONNX
    ========================================================================
    Pipeline:
      AI thread  →  _ocr_src_frame  →  ocr_worker  →  ocr_results  →  AI thread draws

    Key improvements over EasyOCR:
      • Bilateral filter removes ESP32-CAM grain while keeping text edges sharp
      • PaddleOCR ONNX models: no Japanese hallucination in camera noise
      • Confidence gate ≥ 60 % eliminates phantom single-character detections
      • Stability filter: text must appear in 2 consecutive passes (anti-flicker)
    """
    global ocr_results, text_for_ui
    print("[OCR THREAD] PaddleOCR (RapidOCR-ONNX) worker started")

    CONF_THRESHOLD = 0.60   # reject anything below 60 % — stops hallucinated chars
    STABLE_WINDOW  = 2      # must appear in 2 consecutive passes before showing

    hist_texts: list[set[str]]   = []
    hist_data:  list[list[dict]] = []
    last_ocr_ts = 0.0

    def _translate_cached(text: str) -> str:
        """Return cached translation or call googletrans (cached on success).
        Avoids 0.5–2 s HTTP round-trips for text that repeats every OCR pass.
        """
        if text in _translation_cache:
            return _translation_cache[text]
        try:
            tr_obj = translator.translate(text, dest='en')
            result = (
                tr_obj.text
                if tr_obj and tr_obj.text and tr_obj.text.strip() != text.strip()
                else ""
            )
        except Exception:
            result = ""
        if len(_translation_cache) > 1000:   # prevent unbounded growth
            _translation_cache.clear()
        _translation_cache[text] = result
        return result

    while pipeline_running:
        # Only run when Mode 2 is active
        with mode_lock:
            is_mode2 = (current_mode == 2)
        if not is_mode2:
            time.sleep(0.5)
            continue

        # Grab latest source frame — skip if not new
        with _ocr_src_lock:
            src    = _ocr_src_frame[0]
            src_ts = _ocr_src_frame[1]
        if src is None or src_ts <= last_ocr_ts:
            time.sleep(0.05)
            continue
        last_ocr_ts = src_ts

        try:
            # ── Center-square crop — patients hold objects at eye level in the middle
            h, w = src.shape[:2]           # typically 480×640 from ESP32-CAM
            side   = min(h, w)             # square side = shorter dimension
            cy, cx = h // 2, w // 2
            y0 = max(cy - side // 2, 0)
            x0 = max(cx - side // 2, 0)
            cropped = src[y0:y0 + side, x0:x0 + side]

            # ── Unsharp mask — amplifies edge contrast for printed letter strokes
            # Formula: sharpened = original + amount × (original − blurred)
            gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
            sharp   = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)   # amount = 0.5

            # ── RapidOCR inference (PaddleOCR ONNX models) ────────────────
            raw, _ = ocr_engine(sharp)  # type: ignore[misc]

            run_texts: set[str]   = set()
            run_data:  list[dict] = []

            if raw:
                for line in raw:
                    box  = line[0]                   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    text = str(line[1]).strip()
                    conf = float(line[2])

                    if conf < CONF_THRESHOLD or len(text) < 2:
                        continue

                    # Map bbox back to full-frame coordinates (undo center-crop offset)
                    tl = [int(box[0][0]) + x0, int(box[0][1]) + y0]
                    br = [int(box[2][0]) + x0, int(box[2][1]) + y0]

                    # Cached translation — avoids re-hitting Google Translate for repeated text
                    translation = _translate_cached(text)

                    run_texts.add(text)
                    run_data.append({
                        "text":        text,
                        "translation": translation,
                        "confidence":  int(conf * 100),
                        "tl":          tl,
                        "br":          br,
                    })

            # ── Stability filter — require 2 consecutive detections ────────
            hist_texts.append(run_texts)
            hist_data.append(run_data)
            if len(hist_texts) > STABLE_WINDOW:
                hist_texts.pop(0)
                hist_data.pop(0)

            if len(hist_texts) >= STABLE_WINDOW:
                stable_set = hist_texts[0].intersection(*hist_texts[1:])
                stable = [r for r in run_data if r["text"] in stable_set]
            else:
                stable = run_data   # first pass — show immediately

            # ── LLM simplification — fire-and-forget, never blocks ocr_worker ──
            simplified_speech = ""
            if stable:
                combined_raw = " ".join(r["text"] for r in stable)
                # Only submit a new LLM request if the previous one has finished
                if not _llm_pending[0]:
                    _llm_pending[0] = True
                    _bg_pool.submit(_llm_simplify_async, combined_raw)
                    _bg_pool.submit(_mongo_log_ocr, combined_raw, "")
                # Use whatever result the LLM has already computed (may be 1 cycle behind)
                simplified_speech = _llm_last_result[0]
                for r in stable:
                    r["simplified"] = simplified_speech

            # ── Commit results (thread-safe) ───────────────────────────────
            with ocr_lock:
                ocr_results = stable
                text_for_ui = [
                    (f"{r['text']}  \u2192  {r['translation']}" if r["translation"] else r["text"])
                    for r in stable
                ]
                if simplified_speech:
                    text_for_ui.insert(0, f"\U0001f9e0 {simplified_speech}")

            mode_stats[2]["texts_recognized"] = len(stable)
            if stable:
                mode_stats[2]["translations"] += sum(1 for r in stable if r["translation"])

        except Exception as e:
            print(f"[OCR THREAD] Error: {e}")

        time.sleep(OCR_INTERVAL)

def process_ai_thread() -> None:
    """
    THREAD 2: Worker Feeder + TTS
    Reads the latest frame from VideoStream and feeds each inference worker
    thread (YOLO / OCR / emotion / sign / face). Handles TTS alerts.
    Drawing is done inside generate_frames so this thread never touches
    the video pipeline — AI slowness cannot lag the camera feed.
    """
    print("[THREAD 2] Worker Feeder thread started")
    global current_emotion

    last_seen_objects: dict = {}
    last_ocr_text:     str  = ""
    last_emotion:      str  = "Detecting"
    last_seen_faces:   dict = {}

    last_ai_ts    = 0.0
    _AI_MIN_INTV  = 0.2   # cap at 5 Hz for poor WiFi - aggressive throttle
    _last_ai_wall = 0.0
    while pipeline_running:
        # Cheaply check if a new ESP32 frame has arrived without decoding anything.
        if vs is None:
            time.sleep(0.1)
            continue
        with vs.lock:
            ts  = vs.timestamp
            raw = vs.raw_jpeg
        if raw is None or ts <= last_ai_ts:
            time.sleep(0.004)   # no new frame yet — yield CPU and retry fast
            continue
        # Wall-clock throttle: skip decode if workers fed recently (saves ~30 decodes/s → ~12/s)
        _now_w = time.time()
        if _now_w - _last_ai_wall < _AI_MIN_INTV:
            time.sleep(0.008)
            continue
        last_ai_ts    = ts
        _last_ai_wall = _now_w

        # Decode exactly ONCE per genuinely new ESP32 JPEG (was: every 33 ms regardless)
        arr   = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        with mode_lock:
            active_mode = current_mode

        if active_mode == 1:
            if currency_mode:
                # Sub-mode: Currency Reader — feed rupee detector, skip general YOLO TTS
                with _currency_src_lock:
                    _currency_src_frame[0] = frame
                    _currency_src_frame[1] = ts
            else:
                # Normal: general object detection
                with _yolo_src_lock:
                    _yolo_src_frame[0] = frame
                    _yolo_src_frame[1] = ts
                current_time = time.time()
                with _yolo_boxes_lock:
                    boxes_snap = list(_yolo_boxes)
                for box in boxes_snap:
                    obj_name = box["raw"]
                    if current_time - last_seen_objects.get(obj_name, 0) > 5:
                        audio.speak(f"{obj_name} ahead")
                        last_seen_objects[obj_name] = current_time
                        break

        elif active_mode == 2:
            if lecture_mode:
                # Sub-mode: Lecture Logger — feed whiteboard OCR worker
                with _lecture_src_lock:
                    _lecture_src_frame[0] = frame
                    _lecture_src_frame[1] = ts
            else:
                # Normal: general OCR + translation
                with _ocr_src_lock:
                    _ocr_src_frame[0] = frame
                    _ocr_src_frame[1] = ts
                with ocr_lock:
                    results_snap = list(ocr_results)
                if results_snap:
                    simplified = results_snap[0].get("simplified", "")
                    spoken = simplified if simplified else ". ".join(
                        r["translation"] if r["translation"] else r["text"]
                        for r in results_snap
                    )
                    if spoken != last_ocr_text:
                        audio.speak(spoken)
                        last_ocr_text = spoken

        elif active_mode == 3:
            with _emotion_src_lock:
                _emotion_src_frame[0] = frame
                _emotion_src_frame[1] = ts
            with _emotion_cache_lock:
                cached_emotion = _emotion_cache["emotion"]
            current_emotion = cached_emotion
            if cached_emotion not in ("Detecting", "") and cached_emotion != last_emotion:
                audio.speak(f"Subject seems {cached_emotion.lower()}")
                last_emotion = cached_emotion

        elif active_mode == 4:
            with _sign_src_lock:
                _sign_src_frame[0] = frame
                _sign_src_frame[1] = ts

        elif active_mode == 5:
            _last_mode5_frame[0] = frame
            with _face_worker_lock:
                _face_worker_frame[0] = frame
                _face_worker_frame[1] = ts
            current_time = time.time()
            for name in list(_face_names):
                if name != "Unknown" and current_time - last_seen_faces.get(name, 0) > 10:
                    audio.speak(f"{name} approaching")
                    last_seen_faces[name] = current_time
        # No sleep — the vs.timestamp check above is the natural throttle.
        # Loop body is ~0 ms (lock+copy); tight loop is fine and adapts to ESP32 fps.

threading.Thread(target=process_ai_thread,     daemon=True).start()
threading.Thread(target=ocr_worker,            daemon=True).start()
threading.Thread(target=yolo_worker,           daemon=True).start()
threading.Thread(target=currency_worker,       daemon=True).start()  # Mode 1 sub-mode
# threading.Thread(target=lecture_worker,        daemon=True).start()  # DISABLED - Mode 2 sub-mode removed for performance
threading.Thread(target=emotion_worker,        daemon=True).start()
# threading.Thread(target=face_worker,           daemon=True).start()  # DISABLED - Mode 5 removed for performance
threading.Thread(target=sign_worker,           daemon=True).start()
# AudioAlertSystem already started its own daemon thread in __init__

def _has_overlays(mode: int) -> bool:
    """Return True when there are cached AI annotations to draw on the frame.
    When False, generate_frames serves the raw ESP32 JPEG bytes directly
    — zero decode/encode cost, maximum display FPS.
    """
    if mode == 1:
        if currency_mode:
            with _currency_boxes_lock:
                return bool(_currency_boxes)
        with _yolo_boxes_lock:
            return bool(_yolo_boxes)
    elif mode == 2:
        if lecture_mode:
            with _lecture_lock:
                return bool(_lecture_lines)
        with ocr_lock:
            return bool(ocr_results)
    elif mode == 3:
        with _emotion_cache_lock:
            return bool(_emotion_cache["boxes"])
    elif mode == 4:
        with _sign_cache_lock:
            return _sign_cache.get("landmarks") is not None
    elif mode == 5:
        return bool(_face_locations)
    return False


def generate_frames():
    """
    Web Serving — streams annotated MJPEG to the browser.

    Fast path  (no AI detections): serves the raw ESP32 JPEG bytes directly
    — zero decode/encode cost, minimum possible latency.

    Slow path (overlays present): decodes JPEG → draws overlays → re-encodes.
    All drawing is pure OpenCV (<2 ms) so latency stays low.

    Falls back to the last-good frame on WiFi hiccups so the MJPEG connection
    never starves (starvation causes the browser to close the socket, freezing
    the feed permanently).
    """
    print("[STREAM] Web serving started")
    BOUNDARY = b'--frame\r\nContent-Type: image/jpeg\r\n'
    last_raw: Optional[bytes] = None
    last_send_time = 0.0
    KEEPALIVE_FPS = 60   # re-send last frame at 60fps for smooth display

    while pipeline_running:
        # Zero-CPU wait — wakes the instant vs.update() stores a new frame.
        # 0.3s timeout: ESP32 WiFi can jitter 50-200ms; this gives headroom.
        raw, _ts = vs.wait_new_frame(timeout=0.3)
        
        # Keepalive: if no new frame, re-send last_raw at 10fps to keep connection alive
        if raw is None:
            now = time.time()
            if last_raw is not None and (now - last_send_time) >= (1.0 / KEEPALIVE_FPS):
                raw = last_raw
                last_send_time = now
            else:
                time.sleep(0.01)   # brief yield if nothing to send
                continue
        else:
            last_raw = raw
            last_send_time = time.time()

        with mode_lock:
            active_mode = current_mode

        # ── Fast path: no annotations → serve raw ESP32 JPEG (0 ms compute) ────────
        if not _has_overlays(active_mode):
            yield (BOUNDARY +
                   b'Content-Length: ' + str(len(raw)).encode() + b'\r\n\r\n' +
                   raw + b'\r\n')
            continue

        # ── Slow path: decode → draw overlays → re-encode ──────────────────────────
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # ── Apply cached annotation overlays — all pure OpenCV, <1 ms ───────────
        if active_mode == 1:
            if currency_mode:
                # ── Currency Reader overlay ────────────────────────────────────
                with _currency_boxes_lock:
                    cur_snap = list(_currency_boxes)
                if not RUPEE_MODEL_AVAILABLE:
                    cv2.putText(frame, "Rupee model not loaded — run train_rupee_yolo.py",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)
                for box in cur_snap:
                    tl, br = box["tl"], box["br"]
                    denom  = box["denomination"]
                    conf   = box["conf"]
                    cv2.rectangle(frame, tl, br, (0, 215, 255), 3)
                    label = f"{denom}  {conf:.0%}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    ty = max(tl[1] - 8, lh + 8)
                    cv2.rectangle(frame, (tl[0], ty - lh - 6),
                                  (tl[0] + lw + 8, ty + 4), (0, 215, 255), -1)
                    cv2.putText(frame, label, (tl[0] + 4, ty - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                # Sub-mode banner
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 215, 255), -1)
                cv2.putText(frame, "💵 CURRENCY READER — Indian Rupees",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                # ── Normal YOLO overlay ────────────────────────────────────────
                with _yolo_boxes_lock:
                    boxes_snap = list(_yolo_boxes)
                for box in boxes_snap:
                    cv2.rectangle(frame, box["tl"], box["br"], (0, 210, 60), 2)
                    (lw, lh), _ = cv2.getTextSize(box["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    ty = max(box["tl"][1] - 6, lh + 6)
                    cv2.rectangle(frame, (box["tl"][0], ty - lh - 4),
                                  (box["tl"][0] + lw + 6, ty + 2), (0, 210, 60), -1)
                    cv2.putText(frame, box["label"], (box["tl"][0] + 3, ty - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        elif active_mode == 2:
            if lecture_mode:
                # ── Lecture Logger overlay ─────────────────────────────────────
                with _lecture_lock:
                    recent = list(_lecture_lines[-5:])   # last 5 whiteboard lines
                    summary_snap = _lecture_summary[0]
                # Draw bounding boxes for each detected whiteboard text region
                for r in recent:
                    if "tl" in r and "br" in r:
                        tl = tuple(r["tl"]); br = tuple(r["br"])
                        cv2.rectangle(frame, tl, br, (60, 200, 60), 1)
                # Banner at top
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (34, 139, 34), -1)
                cv2.putText(frame, "📝 LECTURE LOGGER",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                # Latest line captured
                if recent:
                    last_text = recent[-1]["text"][:60]
                    cv2.rectangle(frame, (0, 34), (frame.shape[1], 62), (0, 0, 0), -1)
                    cv2.putText(frame, last_text, (8, 54),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
                # One-line summary at bottom
                if summary_snap:
                    first_bullet = summary_snap.split("\n")[0][:70]
                    fh = frame.shape[0]
                    cv2.rectangle(frame, (0, fh - 36), (frame.shape[1], fh), (0, 0, 0), -1)
                    cv2.putText(frame, first_bullet, (8, fh - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
            else:
                # ── Normal OCR overlay ─────────────────────────────────────────
                with ocr_lock:
                    results_snap = list(ocr_results)
                for r in results_snap:
                    tl = tuple(r["tl"])
                    br = tuple(r["br"])
                    cv2.rectangle(frame, tl, br, (0, 210, 255), 2)
                    label = r["text"]
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    ty = max(tl[1] - 6, lh + 6)
                    cv2.rectangle(frame, (tl[0], ty - lh - 4),
                                  (tl[0] + lw + 6, ty + 2), (0, 210, 255), -1)
                    cv2.putText(frame, label, (tl[0] + 3, ty - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        elif active_mode == 3:
            with _emotion_cache_lock:
                cached_boxes   = list(_emotion_cache["boxes"])
                cached_emotion = _emotion_cache["emotion"]
            for (x, y, w, h) in cached_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                if w > 50 and h > 50 and y > 30:
                    ts = cv2.getTextSize(cached_emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, (x, y - 30), (x + ts[0] + 10, y), (0, 165, 255), -1)
                    cv2.putText(frame, cached_emotion, (x + 5, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        elif active_mode == 4:
            with _sign_cache_lock:
                cached_sign = _sign_cache.get("sign", "No Hand Detected")
                cached_lms  = _sign_cache.get("landmarks", None)
            if cached_lms is not None:
                try:
                    h_f, w_f = frame.shape[:2]
                    # Draw connections between landmarks
                    for conn in _HAND_CONS:
                        idx_a, idx_b = conn
                        if idx_a < len(cached_lms) and idx_b < len(cached_lms):
                            lm_a = cached_lms[idx_a]
                            lm_b = cached_lms[idx_b]
                            pa = (int(lm_a.x * w_f), int(lm_a.y * h_f))
                            pb = (int(lm_b.x * w_f), int(lm_b.y * h_f))
                            cv2.line(frame, pa, pb, (0, 180, 255), 2)
                    # Draw landmark dots
                    for lm in cached_lms:
                        px = int(lm.x * w_f)
                        py = int(lm.y * h_f)
                        cv2.circle(frame, (px, py), 4, (0, 255, 180), -1)
                except Exception:
                    pass
            overlay_color = (
                (0, 255, 180) if cached_sign not in ("No Hand Detected", "Model Not Loaded")
                else (0, 80, 255)
            )
            cv2.rectangle(frame, (0, 55), (frame.shape[1], 105), (0, 0, 0), -1)
            cv2.putText(frame, f"Sign: {cached_sign}", (10, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, overlay_color, 2)

        elif active_mode == 5:
            if not FACE_REC_AVAILABLE:
                cv2.putText(frame, "face_recognition not installed",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
                cv2.putText(frame, "Run: pip install face_recognition",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)
            else:
                for (top, right, bottom, left), name in zip(
                        list(_face_locations), list(_face_names)):
                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    color = (0, 255, 120) if name != "Unknown" else (0, 80, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, -1)
                    cv2.putText(frame, name, (left + 6, bottom - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                total_f = len(_face_locations)
                known_f = len([n for n in _face_names if n != "Unknown"])
                hud = f"Faces: {total_f}  Known: {known_f}  DB: {len(known_face_names)}"
                cv2.putText(frame, hud, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 120), 2)

        # Downscale to max 480 px wide — smaller JPEG payload, faster encode, less network
        _h_e, _w_e = frame.shape[:2]
        if _w_e > 480:
            frame = cv2.resize(frame, (480, int(480 * _h_e / _w_e)),
                               interpolation=cv2.INTER_LINEAR)
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ret:
            continue
        jpg_bytes = buf.tobytes()
        yield (BOUNDARY +
               b'Content-Length: ' + str(len(jpg_bytes)).encode() + b'\r\n\r\n' +
               jpg_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control':      'no-cache, no-store, must-revalidate',
            'Pragma':             'no-cache',
            'Expires':            '0',
            'X-Accel-Buffering': 'no',   # disable nginx/proxy buffering
        }
    )


@app.route('/snapshot')
def snapshot():
    """
    Single-frame JPEG endpoint — used by the canvas-based frontend stream.
    Always returns the LATEST frame with ZERO buffer lag (no MJPEG backlog).
    Fast path: no overlays → raw ESP32 JPEG (0 ms decode/encode).
    Slow path: overlays present → decode + draw + re-encode at quality 75.
    """
    with vs.lock:
        raw = vs.raw_jpeg
        raw_ts = vs.timestamp

    if raw is None or time.time() - raw_ts > FRAME_TIMEOUT:
        return Response(status=503)

    with mode_lock:
        active_mode = current_mode

    # ── Fast path ────────────────────────────────────────────────────────────
    if not _has_overlays(active_mode):
        return Response(raw, mimetype='image/jpeg', headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
        })


    # ── Slow path: draw overlays ──────────────────────────────────────────────
    arr   = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return Response(raw, mimetype='image/jpeg', headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
        })

    if active_mode == 1:
        if currency_mode:
            with _currency_boxes_lock:
                cur_snap = list(_currency_boxes)
            if not RUPEE_MODEL_AVAILABLE:
                cv2.putText(frame, "Rupee model not loaded — run train_rupee_yolo.py",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)
            for box in cur_snap:
                tl, br = box["tl"], box["br"]
                denom  = box["denomination"]
                conf   = box["conf"]
                cv2.rectangle(frame, tl, br, (0, 215, 255), 3)
                label = f"{denom}  {conf:.0%}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                ty = max(tl[1] - 8, lh + 8)
                cv2.rectangle(frame, (tl[0], ty - lh - 6),
                              (tl[0] + lw + 8, ty + 4), (0, 215, 255), -1)
                cv2.putText(frame, label, (tl[0] + 4, ty - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 215, 255), -1)
            cv2.putText(frame, "CURRENCY READER — Indian Rupees",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            with _yolo_boxes_lock:
                boxes_snap = list(_yolo_boxes)
            for box in boxes_snap:
                cv2.rectangle(frame, box["tl"], box["br"], (0, 210, 60), 2)
                (lw, lh), _ = cv2.getTextSize(box["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = max(box["tl"][1] - 6, lh + 6)
                cv2.rectangle(frame, (box["tl"][0], ty - lh - 4),
                              (box["tl"][0] + lw + 6, ty + 2), (0, 210, 60), -1)
                cv2.putText(frame, box["label"], (box["tl"][0] + 3, ty - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    elif active_mode == 2:
        if lecture_mode:
            with _lecture_lock:
                recent       = list(_lecture_lines[-5:])
                summary_snap = _lecture_summary[0]
            for r in recent:
                if "tl" in r and "br" in r:
                    cv2.rectangle(frame, tuple(r["tl"]), tuple(r["br"]), (60, 200, 60), 1)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (34, 139, 34), -1)
            cv2.putText(frame, "LECTURE LOGGER",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            if recent:
                cv2.rectangle(frame, (0, 34), (frame.shape[1], 62), (0, 0, 0), -1)
                cv2.putText(frame, recent[-1]["text"][:60], (8, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
        else:
            with ocr_lock:
                results_snap = list(ocr_results)
            for r in results_snap:
                tl = tuple(r["tl"]); br = tuple(r["br"])
                cv2.rectangle(frame, tl, br, (0, 210, 255), 2)
                (lw, lh), _ = cv2.getTextSize(r["text"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = max(tl[1] - 6, lh + 6)
                cv2.rectangle(frame, (tl[0], ty - lh - 4),
                              (tl[0] + lw + 6, ty + 2), (0, 210, 255), -1)
                cv2.putText(frame, r["text"], (tl[0] + 3, ty - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    elif active_mode == 3:
        with _emotion_cache_lock:
            cached_boxes   = list(_emotion_cache["boxes"])
            cached_emotion = _emotion_cache["emotion"]
        for (x, y, w, h) in cached_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            if w > 50 and h > 50 and y > 30:
                ts2 = cv2.getTextSize(cached_emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (x, y - 30), (x + ts2[0] + 10, y), (0, 165, 255), -1)
                cv2.putText(frame, cached_emotion, (x + 5, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    elif active_mode == 4:
        with _sign_cache_lock:
            cached_sign = _sign_cache.get("sign", "No Hand Detected")
            cached_lms  = _sign_cache.get("landmarks", None)
        if cached_lms is not None:
            try:
                h_f, w_f = frame.shape[:2]
                for conn in _HAND_CONS:
                    idx_a, idx_b = conn
                    if idx_a < len(cached_lms) and idx_b < len(cached_lms):
                        lm_a = cached_lms[idx_a]; lm_b = cached_lms[idx_b]
                        cv2.line(frame,
                                 (int(lm_a.x * w_f), int(lm_a.y * h_f)),
                                 (int(lm_b.x * w_f), int(lm_b.y * h_f)),
                                 (0, 180, 255), 2)
                for lm in cached_lms:
                    cv2.circle(frame, (int(lm.x * w_f), int(lm.y * h_f)), 4, (0, 255, 180), -1)
            except Exception:
                pass
        overlay_color = (
            (0, 255, 180) if cached_sign not in ("No Hand Detected", "Model Not Loaded")
            else (0, 80, 255)
        )
        cv2.rectangle(frame, (0, 55), (frame.shape[1], 105), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {cached_sign}", (10, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, overlay_color, 2)

    elif active_mode == 5:
        if not FACE_REC_AVAILABLE:
            cv2.putText(frame, "face_recognition not installed",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
        else:
            for (top, right, bottom, left), name in zip(
                    list(_face_locations), list(_face_names)):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                color = (0, 255, 120) if name != "Unknown" else (0, 80, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, -1)
                cv2.putText(frame, name, (left + 6, bottom - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            hud = f"Faces: {len(_face_locations)}  Known: {len([n for n in _face_names if n != 'Unknown'])}"
            cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 120), 2)

    # Downscale to max 480 px wide before JPEG encode — smaller payload, faster network
    _hs, _ws = frame.shape[:2]
    if _ws > 480:
        frame = cv2.resize(frame, (480, int(480 * _hs / _ws)),
                           interpolation=cv2.INTER_LINEAR)
    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    if not ret:
        return Response(raw, mimetype='image/jpeg', headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
        })

    return Response(buf.tobytes(), mimetype='image/jpeg', headers={
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
    })


@app.route('/set_mode/<int:mode_id>', methods=['POST'])
def set_mode(mode_id):
    global current_mode, latest_ocr_data, text_for_ui, current_emotion
    global currency_mode, lecture_mode
    if 1 <= mode_id <= 7:
        with mode_lock:
            current_mode = mode_id
            # Reset sub-modes on every mode switch
            currency_mode = False
            lecture_mode  = False
            mode_stats[1]["currency_mode"] = False
            mode_stats[2]["lecture_mode"]  = False
            # Clear state when switching modes to avoid stale data
            latest_ocr_data = []
            text_for_ui = []
            current_emotion = "Detecting"
            sign_history.clear()
            _sign_votes.clear()
            face_history.clear()
            last_spoken_face[0] = None
            _face_locations.clear()
            _face_names.clear()
            with _currency_boxes_lock:
                _currency_boxes.clear()
            with _lecture_lock:
                _lecture_lines.clear()
            _lecture_summary[0] = ""
            _currency_last_spoken[0] = ""
            # Clear any queued audio alert when switching modes
            with audio.speech_queue.mutex:
                audio.speech_queue.queue.clear()
        with ocr_lock:
            ocr_results.clear()
        print(f"[Mode Switch] Changed to Mode {current_mode}: {modes_info.get(mode_id, {}).get('name', str(mode_id))}")
        return jsonify({
            "status": "success",
            "mode":   current_mode,
            "info":   modes_info.get(mode_id, {})
        })
    return jsonify({"status": "error", "message": "Invalid mode"})


@app.route('/toggle_currency', methods=['POST'])
def toggle_currency():
    """Enable / disable Currency Reader sub-mode (Mode 1).
    POST body (optional): {"enable": true}  — omit to toggle.
    """
    global currency_mode
    data = request.get_json(force=True, silent=True) or {}
    if "enable" in data:
        currency_mode = bool(data["enable"])
    else:
        currency_mode = not currency_mode
    mode_stats[1]["currency_mode"] = currency_mode
    _currency_last_spoken[0] = ""   # reset debounce so first note is spoken immediately
    with _currency_boxes_lock:
        _currency_boxes.clear()
    state = "enabled" if currency_mode else "disabled"
    print(f"[CURRENCY] Currency Reader {state}")
    return jsonify({
        "status":          "success",
        "currency_mode":   currency_mode,
        "model_available": RUPEE_MODEL_AVAILABLE,
        "message":         f"Currency Reader {state}"
        + ("" if RUPEE_MODEL_AVAILABLE else " (WARNING: rupee model not loaded — run train_rupee_yolo.py)")
    })


@app.route('/toggle_lecture', methods=['POST'])
def toggle_lecture():
    """Enable / disable Lecture Logger sub-mode (Mode 2).
    POST body (optional): {"enable": true, "clear": true}  — clear wipes saved notes.
    """
    global lecture_mode
    data = request.get_json(force=True, silent=True) or {}
    if "enable" in data:
        lecture_mode = bool(data["enable"])
    else:
        lecture_mode = not lecture_mode
    if data.get("clear"):
        with _lecture_lock:
            _lecture_lines.clear()
        _lecture_summary[0] = ""
        _lecture_last_raw[0] = ""
        _lecture_last_summary_ts[0] = 0.0
    mode_stats[2]["lecture_mode"]  = lecture_mode
    mode_stats[2]["lecture_lines"] = len(_lecture_lines)
    state = "enabled" if lecture_mode else "disabled"
    print(f"[LECTURE] Lecture Logger {state}")
    return jsonify({
        "status":        "success",
        "lecture_mode":  lecture_mode,
        "notes_count":   len(_lecture_lines),
        "message":       f"Lecture Logger {state}"
    })


@app.route('/lecture_data')
def lecture_data():
    """Return accumulated lecture notes and latest AI summary.
    Response shape:
      {
        "lines":   [{text, time, conf, tl, br}, ...],  # up to 200 entries
        "summary": "• Point 1\n• Point 2 ...",
        "pending": true/false,                          # LLM summarization running
        "count":   42
      }
    """
    with _lecture_lock:
        lines_snap = list(_lecture_lines)
    return jsonify({
        "lines":   lines_snap,
        "summary": _lecture_summary[0],
        "pending": _lecture_pending[0],
        "count":   len(lines_snap),
    })


@app.route('/currency_data')
def currency_data():
    """Return latest banknote detections and denomination spoken."""
    with _currency_boxes_lock:
        boxes_snap = list(_currency_boxes)
    return jsonify({
        "detections":        boxes_snap,
        "last_denomination": mode_stats[1]["last_denomination"],
        "notes_detected":    mode_stats[1]["notes_detected"],
        "model_available":   RUPEE_MODEL_AVAILABLE,
        "currency_mode":     currency_mode,
    })

@app.route('/mode_info/<int:mode_id>')
def mode_info(mode_id):
    if mode_id in modes_info:
        return jsonify(modes_info[mode_id])
    return jsonify({"error": "Mode not found"})

@app.route('/all_modes')
def all_modes():
    return jsonify(modes_info)

@app.route('/mode_stats/<int:mode_id>')
def mode_stats_endpoint(mode_id):
    if mode_id in mode_stats:
        return jsonify(mode_stats[mode_id])
    return jsonify({"error": "Mode not found"})

@app.route('/current_mode')
def current_mode_endpoint():
    return jsonify({
        "mode": current_mode,
        "info": modes_info[current_mode],
        "stats": mode_stats[current_mode]
    })

@app.route('/ocr_data')
def ocr_data():
    with ocr_lock:
        results_snap = list(ocr_results)
        ui_snap      = list(text_for_ui)
    return jsonify({"text_list": ui_snap, "results": results_snap})

@app.route('/sign_data')
def sign_data_endpoint():
    """Returns the rolling sign prediction history for Mode 4 (Medical Sign Translator)."""
    last = sign_history[-1]["sign"] if sign_history else "None"
    return jsonify({"history": sign_history, "last_sign": last})

@app.route('/face_data')
def face_data_endpoint():
    """Returns rolling face recognition history for Mode 5 (Social Memory)."""
    last = face_history[-1]["name"] if face_history else "None"
    return jsonify({
        "history": face_history[-20:],
        "last_face": last,
        "known_count": len(known_face_names),
        "known_names": known_face_names
    })

# DISABLED: Mode 5 (Face Recognition) removed for performance optimization
# @app.route('/enroll_unknown', methods=['POST'])
# def enroll_unknown():
#     """Hot-enroll an unknown face seen in the current Mode 5 frame.
#     Body: {"name": "Alice"}
#     Saves known_faces/<name>.pkl and reloads into memory — no restart needed.
#     """
#     if not FACE_REC_AVAILABLE:
#         return jsonify({"status": "error", "message": "face_recognition not installed"})
#
#     data = request.get_json(force=True) or {}
#     name = str(data.get("name", "")).strip()
#     if not name:
#         return jsonify({"status": "error", "message": "Name is required"})
#
#     frame = _last_mode5_frame[0]
#     if frame is None:
#         return jsonify({"status": "error", "message": "No frame captured yet — switch to Mode 5 first"})
#
#     # Use the same 25% scale as the AI thread
#     small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
#     locs  = face_recognition.face_locations(rgb, model="hog")
#     encs  = face_recognition.face_encodings(rgb, locs)
#
#     if not encs:
#         return jsonify({"status": "error", "message": "No face detected in frame — move closer and try again"})
#
#     # Pick the first face that is currently Unknown
#     unknown_enc = None
#     for enc in encs:
#         if not known_face_encodings:
#             unknown_enc = enc
#             break
#         matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=0.5)
#         if True not in matches:
#             unknown_enc = enc
#             break
#
#     if unknown_enc is None:
#         return jsonify({"status": "error", "message": "All faces in frame are already enrolled"})
#
#     # Save to disk
#     safe_name = name.replace(" ", "_")
#     out_path  = os.path.join("known_faces", f"{safe_name}.pkl")
#     with open(out_path, "wb") as f:
#         pickle.dump({"encoding": unknown_enc, "name": name}, f)
#
#     # Hot-reload into memory — takes effect on the very next recognition cycle
#     known_face_encodings.append(unknown_enc)
#     known_face_names.append(name)
#     mode_stats[5]["known_faces"] = list(known_face_names)
#
#     print(f"[Mode 5] Enrolled new face: '{name}' → {out_path}")
#     return jsonify({"status": "success", "message": f"Enrolled '{name}' — will be recognised immediately"})

@app.route('/db_log')
def db_log():
    """Return the last 20 records from each MongoDB collection for the UI dashboard."""
    if not MONGO_AVAILABLE:
        return jsonify({"status": "unavailable", "message": "MongoDB not connected"})
    try:
        faces = list(db_faces.find({}, {"_id": 0}).sort("unix_ts", -1).limit(20))  # type: ignore[union-attr]
        signs = list(db_signs.find({}, {"_id": 0}).sort("unix_ts", -1).limit(20))  # type: ignore[union-attr]
        ocr   = list(db_ocr.find({},   {"_id": 0}).sort("unix_ts", -1).limit(20))  # type: ignore[union-attr]
        return jsonify({"status": "ok", "faces": faces, "signs": signs, "ocr": ocr})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/voice_data')
def voice_data():
    """Polled by React UI every 500 ms to update the voice indicator.
    Returns the current voice assistant state: listening, active, last command,
    last response, and whether the wake word just fired.
    """
    with _voice_lock:
        snap = dict(_voice_state)
    return jsonify(snap)


@app.route('/toggle_voice', methods=['POST'])
def toggle_voice():
    """Pause / resume the voice assistant's listening loop.
    POST body (optional): {"enable": true}
    When disabled the listen thread keeps running but commands are dropped
    after transcription (avoids needing to restart the thread).
    """
    data = request.get_json(force=True, silent=True) or {}
    if "enable" in data:
        new_state = bool(data["enable"])
    else:
        with _voice_lock:
            new_state = not _voice_state.get("enabled", False)
    with _voice_lock:
        _voice_state["enabled"] = new_state
    print(f"[VOICE] {'Enabled' if new_state else 'Disabled'} via API")
    return jsonify({"status": "success", "voice_enabled": new_state,
                    "whisper_available": WHISPER_AVAILABLE,
                    "sr_available": SR_AVAILABLE})


@app.route('/health')
def health():
    with _voice_lock:
        va_snap = dict(_voice_state)
    return jsonify({
        "status":          "online",
        "current_mode":    current_mode,
        "mode_name":       modes_info.get(current_mode, {}).get("name", str(current_mode)),
        "device":          "cuda" if torch.cuda.is_available() else "cpu",
        "mongodb":         "connected" if MONGO_AVAILABLE else "offline",
        "ollama":          f"connected ({OLLAMA_MODEL})" if OLLAMA_AVAILABLE else "offline",
        "currency_mode":   currency_mode,
        "rupee_model":     "loaded" if RUPEE_MODEL_AVAILABLE else "not loaded",
        "lecture_mode":    lecture_mode,
        "lecture_notes":   len(_lecture_lines),
        "voice_assistant": {
            "enabled":   va_snap.get("enabled", False),
            "listening": va_snap.get("listening", False),
            "whisper":   WHISPER_AVAILABLE,
            "sr":        SR_AVAILABLE,
        },
    })

if __name__ == "__main__":
    print("=" * 60)
    print("ZYGLASS Vision Platform v2.1 - Multi-threaded Pipeline")
    print("=" * 60)
    print(f"[STARTUP] Running on 0.0.0.0:5000")
    print(f"[SYSTEM] Available modes: {list(modes_info.keys())}")
    print(f"[DEVICE] Using {device}")
    print(f"[MONGO]  {'Connected → AIGLASS database' if MONGO_AVAILABLE else 'Offline — start MongoDB to enable logging'}")
    print(f"[LLM]    {'Ollama + ' + OLLAMA_MODEL + ' ready' if OLLAMA_AVAILABLE else 'Offline — run: ollama serve && ollama pull llama3.2'}")
    print("[PIPELINE] Architecture:")
    print("  Thread 1: Frame Capture (camera -> capture_queue)")
    print("  Thread 2: AI Processing (capture_queue -> processing -> display_queue)")
    print("  Thread 3: Web Serving (display_queue -> clients)")
    print("=" * 60)
    try:
        from werkzeug.serving import WSGIRequestHandler  # type: ignore
        WSGIRequestHandler.wbufsize = 0  # flush each MJPEG chunk immediately — no Nagle lag
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pipeline_running = False
        print("\n[SHUTDOWN] Stopping pipeline threads...")
        if vs is not None:
            vs.stop()
        print("[SHUTDOWN] Pipeline stopped")