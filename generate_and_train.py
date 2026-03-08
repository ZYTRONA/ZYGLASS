#!/usr/bin/env python3
# cspell:ignore landmarker joblib sklearn randint
"""
generate_and_train.py  –  v2  (improved poses + per-finger augmentation)
=========================================================================
Generates anatomically realistic synthetic hand-landmark data for 5
medical signs, then trains a Random Forest + Gradient Boosting ensemble
→  asl_model.pkl   (loaded automatically by backend.py)

KEY IMPROVEMENTS OVER v1
─────────────────────────
• Canonical poses redesigned with large inter-class separation:
    - Help  : all 5 fingers fully spread (wide open palm)
    - Pain  : ONLY index pointing up; all others tightly curl INTO palm
    - Yes   : compact closed fist; all tips close to MCP knuckles
    - No    : index + middle up AND spread apart (V); ring+pinky deep-curled
    - Water : index + middle + RING up (W); ONLY pinky curled
              (ring tip landmark 16 is the key feature separating Water from No)
• Per-finger independent articulation noise
• Left-hand mirror augmentation doubles effective samples
• Soft Voting ensemble: RandomForest (200) + GradientBoosting (150)
• 500 samples per sign (× 2 mirrored = 1000 effective)
"""

import csv
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

OUTPUT_CSV       = "medical_signs.csv"
MODEL_OUTPUT     = "asl_model.pkl"
SAMPLES_PER_SIGN = 500
NOISE_STD        = 0.020
SCALE_RANGE      = (0.75, 1.25)
TRANS_RANGE      = 0.12
ROT_RANGE        = 0.25
FINGER_JITTER    = 0.015
RNG_SEED         = 42
SIGNS = ["Help", "Pain", "Yes", "No", "Water"]

CANONICAL_POSES = {
    # HELP: all 5 fingers wide open, spread laterally
    "Help": [
        (0.50, 0.85, 0.00),
        (0.34, 0.78, 0.01), (0.24, 0.71, 0.01), (0.15, 0.63, 0.01), (0.08, 0.56, 0.01),
        (0.38, 0.62, 0.00), (0.36, 0.47, 0.00), (0.35, 0.37, 0.00), (0.34, 0.27, 0.00),
        (0.50, 0.60, 0.00), (0.50, 0.44, 0.00), (0.50, 0.34, 0.00), (0.50, 0.24, 0.00),
        (0.62, 0.62, 0.00), (0.64, 0.47, 0.00), (0.65, 0.37, 0.00), (0.66, 0.27, 0.00),
        (0.72, 0.68, 0.00), (0.76, 0.56, 0.00), (0.78, 0.47, 0.00), (0.80, 0.40, 0.00),
    ],
    # PAIN: ONLY index up; all others deeply curled into palm
    "Pain": [
        (0.50, 0.85, 0.00),
        (0.41, 0.78, 0.04), (0.44, 0.74, 0.07), (0.47, 0.72, 0.09), (0.49, 0.70, 0.09),
        (0.39, 0.63, 0.00), (0.38, 0.48, 0.00), (0.37, 0.38, 0.00), (0.37, 0.28, 0.00),
        (0.51, 0.62, 0.00), (0.52, 0.69, 0.07), (0.52, 0.74, 0.10), (0.52, 0.76, 0.10),
        (0.61, 0.64, 0.00), (0.62, 0.71, 0.07), (0.62, 0.75, 0.10), (0.62, 0.77, 0.10),
        (0.69, 0.68, 0.00), (0.70, 0.74, 0.06), (0.71, 0.78, 0.08), (0.71, 0.80, 0.08),
    ],
    # YES: tight closed fist, ALL fingers curled
    "Yes": [
        (0.50, 0.85, 0.00),
        (0.40, 0.79, 0.02), (0.41, 0.73, 0.04), (0.44, 0.68, 0.06), (0.47, 0.65, 0.06),
        (0.41, 0.63, 0.00), (0.42, 0.69, 0.07), (0.43, 0.73, 0.10), (0.43, 0.75, 0.10),
        (0.51, 0.62, 0.00), (0.51, 0.68, 0.07), (0.51, 0.73, 0.10), (0.51, 0.75, 0.10),
        (0.59, 0.63, 0.00), (0.60, 0.69, 0.07), (0.60, 0.74, 0.10), (0.60, 0.76, 0.10),
        (0.67, 0.68, 0.00), (0.68, 0.73, 0.06), (0.68, 0.77, 0.08), (0.68, 0.79, 0.08),
    ],
    # NO: ONLY index + middle extended & spread wide (V sign); ring + pinky deeply curled
    "No": [
        (0.50, 0.85, 0.00),
        (0.39, 0.78, 0.03), (0.37, 0.73, 0.05), (0.39, 0.70, 0.06), (0.42, 0.67, 0.06),
        (0.38, 0.63, 0.00), (0.35, 0.48, 0.00), (0.33, 0.37, 0.00), (0.32, 0.27, 0.00),
        (0.51, 0.61, 0.00), (0.54, 0.46, 0.00), (0.56, 0.35, 0.00), (0.57, 0.25, 0.00),
        (0.61, 0.63, 0.00), (0.62, 0.70, 0.07), (0.63, 0.75, 0.10), (0.63, 0.77, 0.10),
        (0.69, 0.68, 0.00), (0.70, 0.74, 0.06), (0.71, 0.78, 0.08), (0.71, 0.80, 0.08),
    ],
    # WATER: index + middle + RING all extended (W); ONLY pinky curled
    # Ring tip (landmark 16) at y≈0.26 distinguishes Water from No (where ring tip y≈0.77)
    "Water": [
        (0.50, 0.85, 0.00),
        (0.38, 0.78, 0.03), (0.37, 0.73, 0.05), (0.40, 0.70, 0.06), (0.44, 0.67, 0.06),
        (0.38, 0.63, 0.00), (0.36, 0.48, 0.00), (0.35, 0.37, 0.00), (0.34, 0.27, 0.00),
        (0.50, 0.61, 0.00), (0.50, 0.46, 0.00), (0.50, 0.35, 0.00), (0.50, 0.25, 0.00),
        (0.62, 0.62, 0.00), (0.64, 0.47, 0.00), (0.65, 0.36, 0.00), (0.66, 0.26, 0.00),
        (0.70, 0.68, 0.00), (0.71, 0.74, 0.06), (0.72, 0.78, 0.09), (0.72, 0.80, 0.09),
    ],
}

_FINGERS = [
    [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20],
]

def normalize_landmarks(raw_63):
    pts = np.array(raw_63, dtype=np.float32).reshape(21, 3)
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[9]))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten().tolist()

def augment(pose, n, rng):
    base = np.array(pose, dtype=np.float32)
    samples = []
    for _ in range(n):
        p = base.copy()
        scale = float(rng.uniform(*SCALE_RANGE))
        p[:, :2] = p[0, :2] + (p[:, :2] - p[0, :2]) * scale
        p[:, 0] += float(rng.uniform(-TRANS_RANGE, TRANS_RANGE))
        p[:, 1] += float(rng.uniform(-TRANS_RANGE, TRANS_RANGE))
        angle = float(rng.uniform(-ROT_RANGE, ROT_RANGE))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        cx, cy = p[0, 0], p[0, 1]
        dx, dy = p[:, 0] - cx, p[:, 1] - cy
        p[:, 0] = cx + cos_a * dx - sin_a * dy
        p[:, 1] = cy + sin_a * dx + cos_a * dy
        for finger_lms in _FINGERS:
            fy = float(rng.uniform(-FINGER_JITTER * 3, FINGER_JITTER * 3))
            fz = float(rng.uniform(-FINGER_JITTER * 2, FINGER_JITTER * 2))
            for lm_idx in finger_lms[1:]:
                p[lm_idx, 1] += fy
                p[lm_idx, 2] += fz
        p += rng.normal(0.0, NOISE_STD, p.shape).astype(np.float32)
        samples.append(normalize_landmarks(p.flatten().tolist()))
    return samples

def mirror_sample(row_63):
    pts = np.array(row_63, dtype=np.float32).reshape(21, 3)
    pts[:, 0] = -pts[:, 0]
    return pts.flatten().tolist()

def main():
    rng = np.random.default_rng(RNG_SEED)
    print("=" * 60)
    print("  Medical Sign Generator + Trainer  v2")
    print("=" * 60)
    print(f"\n[DATA] {SAMPLES_PER_SIGN} × 2 (mirrored) × {len(SIGNS)} signs …\n")

    headers = [f"{c}{i}" for i in range(21) for c in ["x", "y", "z"]] + ["Label"]
    all_rows = []
    for sign in SIGNS:
        samples = augment(CANONICAL_POSES[sign], SAMPLES_PER_SIGN, rng)
        mirrored = [mirror_sample(s) for s in samples]
        for s in samples:
            all_rows.append(s + [sign])
        for s in mirrored:
            all_rows.append(s + [sign])
        print(f"  ✓  {sign:<10s}  {len(samples)+len(mirrored)} samples")

    rng.shuffle(all_rows)
    with open(OUTPUT_CSV, "w", newline="") as fh:
        csv.writer(fh).writerow(headers)
        csv.writer(fh).writerows(all_rows)
    print(f"\n[DATA] {len(all_rows)} rows → {OUTPUT_CSV}")

    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)
    feat = [c for c in df.columns if c != "Label"]
    X = df[feat].to_numpy(dtype=np.float32)
    y = np.asarray(df["Label"].to_numpy(dtype=str))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RNG_SEED)
    print(f"[TRAIN] Train: {len(X_tr)}   Test: {len(X_te)}")

    print("[TRAIN] Fitting ensemble (RF + GB) …")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RNG_SEED)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=RNG_SEED)
    ensemble = VotingClassifier([("rf", rf), ("gb", gb)], voting="soft", n_jobs=-1)
    ensemble.fit(X_tr, y_tr)

    y_pred = ensemble.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cv = cross_val_score(ensemble, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"\n{'─'*52}")
    print(f"  Test accuracy : {acc*100:.1f}%")
    print(f"  5-fold CV     : {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")
    print(f"{'─'*52}\n")
    print(classification_report(y_te, y_pred, zero_division=0))

    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y_te, y_pred, labels=labels_sorted)
    print("Confusion matrix (row=true, col=pred):")
    print(f"{'':>8s}" + "".join(f"{l:>8s}" for l in labels_sorted))
    for i, rl in enumerate(labels_sorted):
        print(f"{rl:>8s}" + "".join(f"{cm[i,j]:>8d}" for j in range(len(labels_sorted))))

    joblib.dump(ensemble, MODEL_OUTPUT, compress=3)
    print(f"\n[SAVED] {MODEL_OUTPUT}  ({os.path.getsize(MODEL_OUTPUT)//1024} KB)")
    print("\n[NEXT]  Restart backend.py — Mode 4 will auto-load asl_model.pkl")
    print("\n  Signs to use:")
    print("    Help  = open palm, all 5 fingers spread")
    print("    Pain  = index finger only, pointing up")
    print("    Yes   = closed fist")
    print("    No    = V sign  (index + middle, 2 fingers)")
    print("    Water = W sign  (index + middle + RING, 3 fingers)\n")

if __name__ == "__main__":
    main()
