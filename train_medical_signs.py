#!/usr/bin/env python3
"""
Medical Sign Language Model Trainer — ZYGLASS v2
==================================================
Reads  medical_signs.csv  (built by collect_medical_signs.py) and trains a
soft-voting ensemble of Random Forest + Gradient Boosting classifiers that maps
63 hand-landmark coordinates to a medical word label.

Improvements over v1:
  • Wrist-centred normalisation  (landmarks ÷ wrist→middle-tip distance, lm 12)
  • XY-plane rotation augmentation at ±5° and ±10°   (5× data expansion)
  • GradientBoostingClassifier added alongside RandomForest (soft-vote ensemble)
  • _WrappedModel: model.predict() returns raw string labels directly

Output
------
  asl_model.pkl  — drop into project root; backend.py Mode 4 loads it.

Steps
-----
  1. Ensure medical_signs.csv exists (run collect_medical_signs.py first).
  2. Activate the venv:  source train_env/bin/activate
  3. Run:  python train_medical_signs.py
  4. Restart backend.py — Mode 4 loads the model automatically.
"""

import math
import os
import sys

import joblib          # type: ignore
import numpy as np     # type: ignore
import pandas as pd    # type: ignore
from sklearn.ensemble import (  # type: ignore
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report  # type: ignore
from sklearn.model_selection import cross_val_score, train_test_split  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

# ── Configuration ────────────────────────────────────────────────────────────
CSV_FILE         = "medical_signs.csv"
OUTPUT_MODEL     = "asl_model.pkl"
TEST_SPLIT       = 0.20              # 20 % held-out test set
CV_FOLDS         = 5                 # k in k-fold cross-validation
RANDOM_STATE     = 42
MIN_SAMPLES_WARN = 50                # Warn if any class has fewer samples
AUG_ANGLES       = [5, -5, 10, -10] # XY rotation augmentation angles (degrees)
# ─────────────────────────────────────────────────────────────────────────────


# ─── Landmark helpers ─────────────────────────────────────────────────────────

def normalize(pts: np.ndarray) -> np.ndarray:
    """Wrist-centred normalisation: subtract wrist (lm 0), scale by wrist→middle-MCP (lm 9).

    Kept as a reference / standalone utility.  The new collect_medical_signs.py
    already applies this normalisation at collection time, so augment() no longer
    calls this function — it receives pre-normalised data from the CSV.

    Args:
        pts: (21, 3) float32 raw hand landmarks.

    Returns:
        (21, 3) normalised array.
    """
    pts = pts.copy().astype(np.float32)
    pts -= pts[0]                          # centre on wrist
    scale = float(np.linalg.norm(pts[9])) # wrist → middle-finger MCP (lm 9)
    if scale > 1e-6:
        pts /= scale
    return pts


def rotate_xy(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate landmark XY coordinates by *angle_deg* degrees; Z unchanged.

    Args:
        pts:       (21, 3) float32 normalised landmarks.
        angle_deg: Rotation angle in degrees (positive = counter-clockwise).

    Returns:
        (21, 3) rotated landmarks.
    """
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0., 0., 1.0]], dtype=np.float32)
    return (R @ pts.T).T   # (21, 3)


def augment(raw63: np.ndarray) -> list:
    """Return original + rotation-augmented variants as flat (63,) arrays.

    The updated collect_medical_signs.py already applies wrist-centred /
    lm9-scaled normalisation at collection time, so the CSV rows are already
    normalised.  This function only applies XY rotation augmentation.

    Args:
        raw63: (63,) pre-normalised landmark floats from the CSV.

    Returns:
        List of 5 augmented (63,) arrays  [orig, +5°, −5°, +10°, −10°].
    """
    pts = raw63.reshape(21, 3)   # already normalised at collection time
    variants = [pts.flatten()]
    for angle in AUG_ANGLES:
        variants.append(rotate_xy(pts, angle).flatten())
    return variants


# ─── Wrapped model ────────────────────────────────────────────────────────────

class _WrappedModel:
    """Thin wrapper so model.predict() returns raw string labels directly.

    backend.py Mode 4 calls  model.predict([features])  and expects a string.
    This class handles the LabelEncoder inverse_transform transparently.
    """

    def __init__(self, ensemble: VotingClassifier, le: LabelEncoder) -> None:
        self._ensemble = ensemble
        self._le       = le

    def predict(self, X: np.ndarray) -> np.ndarray:
        encoded = self._ensemble.predict(np.asarray(X, dtype=np.float32))
        return self._le.inverse_transform(encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._ensemble.predict_proba(np.asarray(X, dtype=np.float32))

    def __repr__(self) -> str:
        return f"_WrappedModel(ensemble={self._ensemble!r})"


# ─── Pretty output ────────────────────────────────────────────────────────────

def _print_banner(text: str) -> None:
    border = "─" * len(text)
    print(f"\n┌{border}┐")
    print(f"│{text}│")
    print(f"└{border}┘")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _print_banner("  ZYGLASS Medical Sign Language Trainer v2  ")

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] {CSV_FILE} not found.")
        print("        Run collect_medical_signs.py first to build the dataset.")
        sys.exit(1)

    print(f"\n[LOAD]  Reading {CSV_FILE} …")
    df = pd.read_csv(CSV_FILE)

    if df.empty:
        print("[ERROR] CSV file is empty.")
        sys.exit(1)
    if "Label" not in df.columns:
        print("[ERROR] 'Label' column missing from CSV.")
        sys.exit(1)

    expected_feature_cols = 63   # 21 joints × 3 axes
    feature_cols = [c for c in df.columns if c != "Label"]
    if len(feature_cols) != expected_feature_cols:
        print(
            f"[ERROR] Expected {expected_feature_cols} feature columns, "
            f"found {len(feature_cols)}."
        )
        sys.exit(1)

    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["Label"].to_numpy(dtype=str)
    labels    = sorted(df["Label"].unique())
    n_classes = len(labels)
    print(f"[INFO]  {len(df)} samples  |  {n_classes} classes  |  {X_raw.shape[1]} features")

    # ── 2. Class distribution ─────────────────────────────────────────────────
    print("\n[DIST]  Samples per sign (before augmentation):")
    any_low = False
    for label in labels:
        count = int((y_raw == label).sum())
        bar   = "█" * min(count // 5, 20)
        flag  = "  ← low" if count < MIN_SAMPLES_WARN else ""
        print(f"        {label:<14s}  {count:>4d}  {bar}{flag}")
        if count < MIN_SAMPLES_WARN:
            any_low = True

    if any_low:
        print(
            f"\n[WARN]  Some classes have < {MIN_SAMPLES_WARN} samples. "
            "Augmentation will help, but collecting more real data is recommended."
        )

    # ── 3. Augment ────────────────────────────────────────────────────────────
    print(f"\n[AUG]   Applying XY rotation augmentation at {AUG_ANGLES}° …")
    X_aug_list: list = []
    y_aug_list: list = []

    for raw63, label in zip(X_raw, y_raw):
        variants = augment(raw63)
        X_aug_list.extend(variants)
        y_aug_list.extend([label] * len(variants))

    X_aug = np.array(X_aug_list, dtype=np.float32)
    y_aug = np.array(y_aug_list, dtype=str)
    print(
        f"[AUG]   {len(df)} → {len(X_aug)} samples  "
        f"(×{len(AUG_ANGLES) + 1} via rotation + normalisation)"
    )

    # ── 4. Label encoding ─────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(labels)
    y_encoded = le.transform(y_aug)

    # ── 5. Train / test split ─────────────────────────────────────────────────
    print(f"\n[SPLIT] {int((1 - TEST_SPLIT) * 100)}% train / {int(TEST_SPLIT * 100)}% test (stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_encoded,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    print(f"        Train: {len(X_train)} samples   Test: {len(X_test)} samples")

    # ── 6. Build RF + GBT ensemble ────────────────────────────────────────────
    print("\n[TRAIN] Building RF + GBT soft-voting ensemble …")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    print("[TRAIN] Done.")

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    y_pred_enc = ensemble.predict(X_test)
    y_pred_str = le.inverse_transform(y_pred_enc)
    y_test_str = le.inverse_transform(y_test)

    acc = accuracy_score(y_test_str, y_pred_str)
    print(f"\n[RESULT]  Test accuracy: {acc * 100:.1f}%")
    print()
    print("[REPORT]")
    print(classification_report(y_test_str, y_pred_str, target_names=labels))

    # ── 8. Cross-validation ───────────────────────────────────────────────────
    print(f"[CV]    {CV_FOLDS}-fold cross-validation on augmented data …")
    cv_scores = cross_val_score(ensemble, X_aug, y_encoded, cv=CV_FOLDS, n_jobs=-1)
    print(
        f"        Mean: {cv_scores.mean() * 100:.1f}%  "
        f"(± {cv_scores.std() * 100:.1f}%)"
    )

    # ── 9. Feature importance (top 10 from RF sub-model) ─────────────────────
    rf_fitted   = ensemble.named_estimators_["rf"]
    importances = rf_fitted.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:10]
    axes        = ["x", "y", "z"]
    print("\n[FEAT]  Top-10 most important landmarks (RF component):")
    for rank, idx in enumerate(top_idx, 1):
        joint = idx // 3
        axis  = axes[idx % 3]
        print(
            f"        #{rank:>2d}  joint {joint:>2d}-{axis}  "
            f"importance={importances[idx]:.4f}"
        )

    # ── 10. Wrap + save ───────────────────────────────────────────────────────
    wrapped = _WrappedModel(ensemble, le)
    joblib.dump(wrapped, OUTPUT_MODEL, compress=3)
    size_kb = os.path.getsize(OUTPUT_MODEL) / 1024
    print(f"\n[SAVED] {OUTPUT_MODEL}  ({size_kb:.1f} KB)")
    print()
    print("Next steps")
    print("──────────")
    print(f"  1. Confirm {OUTPUT_MODEL} is in the project root folder.")
    print("  2. Restart backend.py — Mode 4 loads the model automatically.")
    print("  3. Switch to Mode 4 in the UI and hold a medical sign to test.")
    print()


if __name__ == "__main__":
    main()
