"""
dataset/feature_extractor.py
----------------------------
Feature engineering module for hand gesture classification.
Extracts invariant geometric, distance, and joint-angle features from 21 MediaPipe hand landmarks.

Features extracted:
  1. 63 Wrist-relative, scale-normalized coordinates (21 landmarks x 3D coords).
  2. 14 Pairwise Euclidean distance ratios (fingertip-to-wrist, fingertip-to-MCP, finger spreads).
  3. 5 Joint bend angle features (cosine angle at PIP joints for each finger).
Total feature dimension: 82 continuous numeric features.
"""

import math
import numpy as np


class HandFeatureExtractor:
    """
    Extracts translation-invariant and scale-invariant geometric features
    from 21 hand landmarks for Machine Learning and Deep Learning models.
    """

    FINGER_TIPS = [4, 8, 12, 16, 20]     # Thumb, Index, Middle, Ring, Pinky
    FINGER_PIPS = [3, 6, 10, 14, 18]     # IP/PIP joints
    FINGER_MCPS = [2, 5, 9, 13, 17]      # MCP joints

    def __init__(self):
        self.feature_names = self._generate_feature_names()

    def _generate_feature_names(self):
        """Build column names for tabular datasets."""
        names = []
        # 1. 63 Normalized coordinates
        for i in range(21):
            names.extend([f"norm_x_{i}", f"norm_y_{i}", f"norm_z_{i}"])
        # 2. 5 Tip-to-wrist distances
        for tip in self.FINGER_TIPS:
            names.append(f"dist_wrist_tip_{tip}")
        # 3. 5 Tip-to-MCP distances (curl metric)
        for tip, mcp in zip(self.FINGER_TIPS, self.FINGER_MCPS):
            names.append(f"dist_mcp_tip_{mcp}_{tip}")
        # 4. 4 Finger spread distances
        names.append("dist_thumb_index")
        names.append("dist_index_middle")
        names.append("dist_middle_ring")
        names.append("dist_ring_pinky")
        # 5. 5 Joint bend cosines
        for i in range(5):
            names.append(f"angle_bend_finger_{i}")
        return names

    def extract_features(self, landmarks):
        """
        Extract the 82-dimensional feature vector from 21 landmarks.
        :param landmarks: list of (id, x, y, z) or 21x3 array
        :return: 1D numpy array of shape (82,) or None if invalid
        """
        if landmarks is None or len(landmarks) < 21:
            return None

        # Convert to numpy array of shape (21, 3)
        if isinstance(landmarks[0], (tuple, list)):
            coords = np.array([[lm[1], lm[2], lm[3] if len(lm) > 3 else 0.0] for lm in landmarks], dtype=np.float32)
        else:
            coords = np.array(landmarks, dtype=np.float32)

        # 1. Wrist-relative translation normalization
        wrist = coords[0]
        rel_coords = coords - wrist

        # 2. Scale normalization: use distance between wrist (0) and middle MCP (9)
        palm_size = np.linalg.norm(rel_coords[9])
        if palm_size < 1e-6:
            palm_size = 1.0
        norm_coords = rel_coords / palm_size

        features = []
        # Flatten 63 normalized coords
        features.extend(norm_coords.flatten())

        # 3. Fingertip to wrist distances
        for tip in self.FINGER_TIPS:
            d = np.linalg.norm(norm_coords[tip])
            features.append(d)

        # 4. Fingertip to MCP distances (measures finger curl)
        for tip, mcp in zip(self.FINGER_TIPS, self.FINGER_MCPS):
            d = np.linalg.norm(norm_coords[tip] - norm_coords[mcp])
            features.append(d)

        # 5. Finger spread distances
        features.append(np.linalg.norm(norm_coords[4] - norm_coords[8]))    # Thumb - Index
        features.append(np.linalg.norm(norm_coords[8] - norm_coords[12]))   # Index - Middle
        features.append(np.linalg.norm(norm_coords[12] - norm_coords[16]))  # Middle - Ring
        features.append(np.linalg.norm(norm_coords[16] - norm_coords[20]))  # Ring - Pinky

        # 6. Joint bend angles (cosine of vector from MCP to PIP and PIP to TIP)
        for mcp, pip, tip in zip(self.FINGER_MCPS, self.FINGER_PIPS, self.FINGER_TIPS):
            v1 = norm_coords[pip] - norm_coords[mcp]
            v2 = norm_coords[tip] - norm_coords[pip]
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                # Clip to [-1.0, 1.0] to prevent numerical domain error
                cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            else:
                cos_angle = 1.0
            features.append(cos_angle)

        return np.array(features, dtype=np.float32)
