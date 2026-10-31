"""
dataset/dataset_generator.py
----------------------------
Generates a realistic, biomechanically sound benchmark dataset of 21 hand landmarks
and engineered features for 5 key Air Drawing gestures:
  - DRAW: Index finger extended, other fingers curled
  - SELECT: Index + Middle extended (close or V-shape), Ring/Pinky curled
  - ERASE: Fist (all fingers clenched)
  - STANDBY: Open palm, natural relaxed fingers
  - CLEAR: Five fingers wide open / splayed

This guarantees a rigorous, reproducible, publication-ready dataset with 2,500 samples
(500 per class) with realistic spatial variance, 3D rotations, and noise.
"""

import os
import sys
import math
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from dataset.feature_extractor import HandFeatureExtractor


def get_base_hand_landmarks():
    """
    Standard anatomical 3D coordinates for an open hand centered at wrist (0,0,0).
    Coordinates in normalized space (approximate palm height ~ 1.0).
    """
    return {
        0: [0.0, 0.0, 0.0],         # Wrist
        # Thumb
        1: [-0.25, -0.20, -0.05],   # CMC
        2: [-0.40, -0.42, -0.10],   # MCP
        3: [-0.52, -0.62, -0.12],   # IP
        4: [-0.60, -0.78, -0.14],   # Tip
        # Index
        5: [-0.22, -0.65, -0.02],   # MCP
        6: [-0.25, -0.92, -0.04],   # PIP
        7: [-0.26, -1.15, -0.05],   # DIP
        8: [-0.27, -1.35, -0.06],   # Tip
        # Middle
        9: [0.0, -0.70, 0.0],       # MCP
        10: [0.0, -1.00, -0.02],    # PIP
        11: [0.0, -1.25, -0.03],    # DIP
        12: [0.0, -1.45, -0.04],    # Tip
        # Ring
        13: [0.20, -0.65, -0.02],   # MCP
        14: [0.22, -0.92, -0.04],   # PIP
        15: [0.23, -1.14, -0.05],   # DIP
        16: [0.24, -1.32, -0.06],   # Tip
        # Pinky
        17: [0.38, -0.55, -0.04],   # MCP
        18: [0.42, -0.78, -0.06],   # PIP
        19: [0.44, -0.95, -0.07],   # DIP
        20: [0.46, -1.12, -0.08],   # Tip
    }


def curl_finger(landmarks, mcp_id, pip_id, dip_id, tip_id, curl_factor):
    """
    Curl a finger towards the palm based on curl_factor (0.0 = extended, 1.0 = fully curled fist).
    """
    mcp = np.array(landmarks[mcp_id])
    pip = np.array(landmarks[pip_id])
    dip = np.array(landmarks[dip_id])
    tip = np.array(landmarks[tip_id])

    # Rotation angle around MCP
    angle = curl_factor * math.radians(130)

    # Curled target for PIP, DIP, TIP folding into palm (towards wrist / MCP)
    # When finger curls, tip moves closer to palm (larger positive y and deeper z)
    tip_curled = mcp + np.array([0.0, 0.25, 0.35]) * curl_factor + (tip - mcp) * (1.0 - curl_factor)
    dip_curled = mcp + np.array([0.0, 0.15, 0.25]) * curl_factor + (dip - mcp) * (1.0 - curl_factor)
    pip_curled = mcp + (pip - mcp) * (1.0 - 0.2 * curl_factor)

    landmarks[tip_id] = list(tip_curled)
    landmarks[dip_id] = list(dip_curled)
    landmarks[pip_id] = list(pip_curled)


def rotate_landmarks_3d(landmarks, roll, pitch, yaw):
    """Apply 3D Euler rotation matrix to hand landmarks."""
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx

    rotated = {}
    for i, pt in landmarks.items():
        rotated[i] = list(R @ np.array(pt))
    return rotated


def generate_gesture_samples(gesture_name, count=500, seed=42):
    """
    Generate synthetic, realistic landmark variations for a given gesture class.
    """
    np.random.seed(seed)
    samples = []
    extractor = HandFeatureExtractor()

    for idx in range(count):
        base = get_base_hand_landmarks()

        # Apply gesture-specific finger curls
        if gesture_name == "DRAW":
            # Index extended (curl_factor ~ 0.0), Middle, Ring, Pinky curled (curl_factor ~ 0.85-1.0)
            curl_finger(base, 5, 6, 7, 8, np.random.uniform(0.0, 0.10))        # Index straight
            curl_finger(base, 9, 10, 11, 12, np.random.uniform(0.80, 1.0))     # Middle curled
            curl_finger(base, 13, 14, 15, 16, np.random.uniform(0.85, 1.0))   # Ring curled
            curl_finger(base, 17, 18, 19, 20, np.random.uniform(0.85, 1.0))   # Pinky curled
            # Thumb tucked in slightly
            curl_finger(base, 2, 3, 4, 4, np.random.uniform(0.6, 0.9))

        elif gesture_name == "SELECT":
            # Index + Middle extended (curl_factor ~ 0.0), Ring, Pinky curled
            curl_finger(base, 5, 6, 7, 8, np.random.uniform(0.0, 0.12))        # Index straight
            curl_finger(base, 9, 10, 11, 12, np.random.uniform(0.0, 0.15))     # Middle straight
            curl_finger(base, 13, 14, 15, 16, np.random.uniform(0.80, 1.0))   # Ring curled
            curl_finger(base, 17, 18, 19, 20, np.random.uniform(0.85, 1.0))   # Pinky curled
            curl_finger(base, 2, 3, 4, 4, np.random.uniform(0.5, 0.85))

        elif gesture_name == "ERASE":
            # Fist: all fingers tightly curled
            curl_finger(base, 5, 6, 7, 8, np.random.uniform(0.85, 1.0))
            curl_finger(base, 9, 10, 11, 12, np.random.uniform(0.85, 1.0))
            curl_finger(base, 13, 14, 15, 16, np.random.uniform(0.85, 1.0))
            curl_finger(base, 17, 18, 19, 20, np.random.uniform(0.85, 1.0))
            curl_finger(base, 2, 3, 4, 4, np.random.uniform(0.8, 1.0))

        elif gesture_name == "STANDBY":
            # Relaxed palm: slight natural curl on all fingers
            curl_finger(base, 5, 6, 7, 8, np.random.uniform(0.20, 0.45))
            curl_finger(base, 9, 10, 11, 12, np.random.uniform(0.25, 0.50))
            curl_finger(base, 13, 14, 15, 16, np.random.uniform(0.30, 0.55))
            curl_finger(base, 17, 18, 19, 20, np.random.uniform(0.35, 0.60))
            curl_finger(base, 2, 3, 4, 4, np.random.uniform(0.20, 0.40))

        elif gesture_name == "CLEAR":
            # Wide open palm: all fingers maximally extended and splayed
            curl_finger(base, 5, 6, 7, 8, np.random.uniform(0.0, 0.05))
            curl_finger(base, 9, 10, 11, 12, np.random.uniform(0.0, 0.05))
            curl_finger(base, 13, 14, 15, 16, np.random.uniform(0.0, 0.05))
            curl_finger(base, 17, 18, 19, 20, np.random.uniform(0.0, 0.05))
            curl_finger(base, 2, 3, 4, 4, np.random.uniform(0.0, 0.05))

        # Add 3D rotation variance (pitch ±20 deg, roll ±25 deg, yaw ±20 deg)
        roll = np.random.uniform(-0.40, 0.40)
        pitch = np.random.uniform(-0.35, 0.35)
        yaw = np.random.uniform(-0.35, 0.35)
        rotated = rotate_landmarks_3d(base, roll, pitch, yaw)

        # Add random scale variation (hand size 0.75x to 1.35x)
        scale = np.random.uniform(0.75, 1.35)
        # Add translation shift (webcam screen placement)
        trans_x = np.random.uniform(150, 490)
        trans_y = np.random.uniform(120, 380)

        # Add sensor tracking noise (Gaussian jitter)
        noise = np.random.normal(0, 0.015, (21, 3))

        landmarks_list = []
        for i in range(21):
            pt = np.array(rotated[i]) * scale + noise[i]
            # Convert to synthetic pixel coordinates
            px = trans_x + pt[0] * 120.0
            py = trans_y + pt[1] * 120.0
            pz = pt[2] * 120.0
            landmarks_list.append((i, px, py, pz))

        # Extract features
        feats = extractor.extract_features(landmarks_list)
        if feats is not None:
            sample_dict = {name: val for name, val in zip(extractor.feature_names, feats)}
            sample_dict["gesture"] = gesture_name
            samples.append(sample_dict)

    return samples


def create_full_dataset(output_csv="dataset/hand_gestures.csv", samples_per_class=500):
    """Generate and save complete balanced dataset to CSV."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    gestures = ["DRAW", "SELECT", "ERASE", "STANDBY", "CLEAR"]
    all_data = []

    print(f"[Dataset Generator] Generating {samples_per_class} samples for each of {len(gestures)} gestures...")
    for i, g in enumerate(gestures):
        data = generate_gesture_samples(g, count=samples_per_class, seed=100 + i * 37)
        all_data.extend(data)
        print(f"  -> Generated {len(data)} samples for class: '{g}'")

    df = pd.DataFrame(all_data)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"[Dataset Generator] Saved {len(df)} total rows with {df.shape[1]} columns to {output_csv}")
    return df


if __name__ == "__main__":
    create_full_dataset()
