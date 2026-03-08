#!/usr/bin/env python3
"""
AI Glass Model Improvement Script
Automates training data collection and model enhancement for all modes.
"""

import os
import cv2
import time
import json
import glob
import shutil
import argparse
import numpy as np
from pathlib import Path

def setup_training_directories():
    """Create organized directories for training data."""
    base_dir = Path("datasets/training_samples")
    
    # Create mode-specific directories
    modes = ["object", "ocr", "emotion", "signs", "faces"]
    for mode in modes:
        for subset in ["raw", "processed", "annotations"]:
            dir_path = base_dir / mode / subset
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✓ Training directories created")

def organize_training_data():
    """Move collected training samples to organized structure."""
    base_dir = Path("datasets/training_samples")
    
    # Find all training files
    training_files = glob.glob("datasets/training_samples/training_*.jpg")
    
    for file_path in training_files:
        filename = os.path.basename(file_path)
        
        # Extract mode from filename: training_object_20250226_143052_123.jpg
        parts = filename.split('_')
        if len(parts) >= 3:
            mode = parts[1]  # object, ocr, emotion, etc.
            
            # Move to organized structure
            target_dir = base_dir / mode / "raw"
            target_path = target_dir / filename
            
            if not target_path.exists():
                shutil.move(file_path, target_path)
                print(f"Moved {filename} → {mode}/raw/")

def enhance_training_dataset(mode: str):
    """Apply enhancement filters to training images for better model performance."""
    mode_dir = Path(f"datasets/training_samples/{mode}")
    raw_dir = mode_dir / "raw"
    processed_dir = mode_dir / "processed"
    
    if not raw_dir.exists():
        print(f"No raw data found for {mode} mode")
        return
    
    image_files = list(raw_dir.glob("*.jpg"))
    print(f"Enhancing {len(image_files)} images for {mode} mode...")
    
    for img_path in image_files:
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Apply mode-specific enhancements
            if mode == "ocr":
                # OCR: enhance text contrast and remove noise
                enhanced = enhance_for_ocr(img)
            elif mode == "emotion":
                # Emotion: enhance facial features
                enhanced = enhance_for_emotion(img)
            elif mode == "object":
                # Object detection: general enhancement
                enhanced = enhance_for_object_detection(img)
            elif mode == "signs":
                # Sign language: enhance hand landmarks
                enhanced = enhance_for_signs(img)
            elif mode == "faces":
                # Face recognition: optimize facial features
                enhanced = enhance_for_faces(img)
            else:
                enhanced = img
            
            # Save enhanced version
            output_path = processed_dir / img_path.name
            cv2.imwrite(str(output_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
        except Exception as e:
            print(f"Enhancement failed for {img_path}: {e}")
    
    print(f"✓ Enhanced {mode} training data")

def enhance_for_ocr(img):
    """Enhance image for better OCR performance."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoise while preserving text edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive threshold for better text contrast
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Convert back to 3-channel for consistency
    enhanced = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    return enhanced

def enhance_for_emotion(img):
    """Enhance image for emotion detection."""
    # Enhance contrast and brightness
    enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=15)
    
    # Apply bilateral filter to smooth skin while keeping facial features sharp
    enhanced = cv2.bilateralFilter(enhanced, 15, 80, 80)
    
    return enhanced

def enhance_for_object_detection(img):
    """General enhancement for object detection."""
    # Denoise
    enhanced = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Enhance contrast
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def enhance_for_signs(img):
    """Enhance for sign language detection."""
    # Convert to HSV for better hand detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Enhance saturation for better hand visibility
    hsv[:,:,1] = cv2.multiply(hsv[:,:,1], np.array([1.3], dtype=np.float32))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply bilateral filter
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return enhanced

def enhance_for_faces(img):
    """Enhance for face recognition."""
    # Histogram equalization for better lighting normalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Smooth skin texture while preserving facial features
    enhanced = cv2.bilateralFilter(enhanced, 15, 80, 80)
    
    return enhanced

def generate_training_report():
    """Generate a report on collected training data."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modes": {}
    }
    
    base_dir = Path("datasets/training_samples")
    modes = ["object", "ocr", "emotion", "signs", "faces"]
    
    for mode in modes:
        mode_dir = base_dir / mode
        raw_count = len(list((mode_dir / "raw").glob("*.jpg"))) if (mode_dir / "raw").exists() else 0
        processed_count = len(list((mode_dir / "processed").glob("*.jpg"))) if (mode_dir / "processed").exists() else 0
        
        report["modes"][mode] = {
            "raw_samples": raw_count,
            "processed_samples": processed_count,
            "total_samples": raw_count + processed_count
        }
    
    # Save report
    report_path = base_dir / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n🎯 TRAINING DATA SUMMARY")
    print("=" * 50)
    for mode, stats in report["modes"].items():
        print(f"{mode.upper():>10}: {stats['total_samples']:>3} samples ({stats['raw_samples']} raw, {stats['processed_samples']} processed)")
    
    total_samples = sum(stats['total_samples'] for stats in report["modes"].values())
    print(f"{'TOTAL':>10}: {total_samples:>3} samples")
    print(f"\n📊 Report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="AI Glass Model Improvement")
    parser.add_argument("--setup", action="store_true", help="Setup training directories")
    parser.add_argument("--organize", action="store_true", help="Organize collected training data")
    parser.add_argument("--enhance", choices=["object", "ocr", "emotion", "signs", "faces", "all"], help="Enhance training data for specific mode")
    parser.add_argument("--report", action="store_true", help="Generate training data report")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_training_directories()
    
    if args.organize:
        organize_training_data()
    
    if args.enhance:
        if args.enhance == "all":
            modes = ["object", "ocr", "emotion", "signs", "faces"]
            for mode in modes:
                enhance_training_dataset(mode)
        else:
            enhance_training_dataset(args.enhance)
    
    if args.report:
        generate_training_report()
    
    if not any([args.setup, args.organize, args.enhance, args.report]):
        print("AI Glass Model Improvement")
        print("Usage: python improve_models.py --setup --organize --enhance all --report")
        print("\nQuick start:")
        print("1. python improve_models.py --setup")
        print("2. Collect data using the Training buttons in the web UI")
        print("3. python improve_models.py --organize --enhance all --report")

if __name__ == "__main__":
    import numpy as np
    main()