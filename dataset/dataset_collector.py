"""
dataset/dataset_collector.py
----------------------------
Interactive real-time hand gesture dataset collector using Webcam + MediaPipe.
Allows the user, researcher, or evaluator to capture customized hand gesture samples live.

Controls:
  - Press [1] or [d]: Record 'DRAW' sample
  - Press [2] or [s]: Record 'SELECT' sample
  - Press [3] or [e]: Record 'ERASE' sample
  - Press [4] or [p]: Record 'STANDBY' sample
  - Press [5] or [c]: Record 'CLEAR' sample
  - Press [q]: Save collected samples and exit
"""

import os
import sys
import time
import cv2
import pandas as pd
import numpy as np

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_tracking import HandTracker
from dataset.feature_extractor import HandFeatureExtractor


def run_collector(output_csv="dataset/hand_gestures_custom.csv"):
    tracker = HandTracker(max_num_hands=1)
    extractor = HandFeatureExtractor()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Collector Error] Could not access webcam.")
        return

    collected_data = []
    class_counts = {"DRAW": 0, "SELECT": 0, "ERASE": 0, "STANDBY": 0, "CLEAR": 0}

    print("\n" + "=" * 50)
    print("  GESTURE DATASET COLLECTOR")
    print("=" * 50)
    print("Hold your hand in front of the camera and press:")
    print("  [1] or [D] -> Record DRAW gesture (Index up)")
    print("  [2] or [S] -> Record SELECT gesture (Index+Middle up)")
    print("  [3] or [E] -> Record ERASE gesture (Closed fist)")
    print("  [4] or [P] -> Record STANDBY gesture (Open palm)")
    print("  [5] or [C] -> Record CLEAR gesture (Splayed 5 fingers)")
    print("  [Q]        -> Finish and save dataset")
    print("=" * 50 + "\n")

    last_record_time = 0
    toast_msg = "Ready to record. Press 1-5 or D/S/E/P/C"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        landmarks = tracker.find_landmarks(frame, draw=True)

        # Draw HUD Box
        cv2.rectangle(frame, (10, 10), (320, 210), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (320, 210), (100, 100, 100), 2)
        cv2.putText(frame, "DATASET RECORDER", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        y_offset = 65
        for g_name, count in class_counts.items():
            cv2.putText(frame, f"{g_name}: {count} samples", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            y_offset += 25

        cv2.putText(frame, toast_msg, (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 80), 1)

        cv2.imshow("Air Drawing - Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        target_gesture = None
        if key in [ord('1'), ord('d'), ord('D')]:
            target_gesture = "DRAW"
        elif key in [ord('2'), ord('s'), ord('S')]:
            target_gesture = "SELECT"
        elif key in [ord('3'), ord('e'), ord('E')]:
            target_gesture = "ERASE"
        elif key in [ord('4'), ord('p'), ord('P')]:
            target_gesture = "STANDBY"
        elif key in [ord('5'), ord('c'), ord('C')]:
            target_gesture = "CLEAR"
        elif key in [ord('q'), ord('Q'), 27]:
            break

        if target_gesture:
            if landmarks and len(landmarks) >= 21:
                feats = extractor.extract_features(landmarks)
                if feats is not None:
                    row = {col: val for col, val in zip(extractor.feature_names, feats)}
                    row["gesture"] = target_gesture
                    row["timestamp"] = time.time()
                    collected_data.append(row)
                    class_counts[target_gesture] += 1
                    toast_msg = f"Captured {target_gesture}! (Total: {class_counts[target_gesture]})"
            else:
                toast_msg = "No hand detected! Bring hand into frame."

    cap.release()
    cv2.destroyAllWindows()

    if collected_data:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df = pd.DataFrame(collected_data)
        if os.path.exists(output_csv):
            df.to_csv(output_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(output_csv, index=False)
        print(f"\n[Collector] Successfully saved {len(df)} samples to {output_csv}!")
    else:
        print("\n[Collector] No samples were captured.")


if __name__ == "__main__":
    run_collector()
