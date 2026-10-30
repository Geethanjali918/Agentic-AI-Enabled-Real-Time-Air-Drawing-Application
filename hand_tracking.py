"""
hand_tracking.py
----------------
Robust hand detection and landmark extraction module using Google MediaPipe.
Supports both modern MediaPipe Tasks (Python 3.12+) and legacy MediaPipe Solutions (Python 3.8-3.11).
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp


class HandTracker:
    """
    Tracks human hands and extracts 21 skeletal landmarks in real-time.
    """

    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6):
        """
        Initialize the hand tracker.
        :param max_num_hands: Maximum number of hands to detect (default: 1 for air drawing)
        :param min_detection_confidence: Confidence threshold for hand detection
        :param min_tracking_confidence: Confidence threshold for landmark tracking
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Determine which MediaPipe API is available in this environment
        self.use_legacy_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

        if self.use_legacy_solutions:
            # Legacy MediaPipe Solutions (Python 3.8 - 3.11)
            self.mp_hands = mp.solutions.hands
            self.detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.mp_draw = mp.solutions.drawing_utils
        else:
            # MediaPipe Tasks Vision API (Python 3.12+)
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_filename = "hand_landmarker.task"
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_filename)

            # Auto-download model weights if not locally present
            if not os.path.exists(model_path):
                print(f"[HandTracker] Downloading MediaPipe model to {model_path}...")
                model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                urllib.request.urlretrieve(model_url, model_path)
                print("[HandTracker] Model downloaded successfully.")

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.detector = mp_vision.HandLandmarker.create_from_options(options)

        # Standard connections between landmarks (21 points)
        self.CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                # Palm base
        ]

    def find_hand_landmarks(self, frame, draw=True):
        """
        Process a BGR frame, locate hand landmarks, and optionally draw skeletal lines.
        :param frame: BGR image from OpenCV
        :param draw: Whether to draw landmark connections on the frame
        :return: (landmarks_list, frame)
                 landmarks_list is a list of (id, x_pixel, y_pixel, z) for 21 points,
                 or empty list if no hand is detected.
        """
        h, w, _ = frame.shape
        landmarks_list = []

        # Convert OpenCV BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.use_legacy_solutions:
            results = self.detector.process(rgb_frame)
            if results.multi_hand_landmarks:
                first_hand = results.multi_hand_landmarks[0]
                for idx, lm in enumerate(first_hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks_list.append((idx, cx, cy, lm.z))

                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        first_hand,
                        self.mp_hands.HAND_CONNECTIONS
                    )
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)

            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                first_hand = results.hand_landmarks[0]
                for idx, lm in enumerate(first_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks_list.append((idx, cx, cy, lm.z))

                if draw:
                    self._draw_landmarks(frame, landmarks_list)

        return landmarks_list, frame

    def _draw_landmarks(self, frame, landmarks):
        """
        Draw clean, subtle hand landmarks and connecting bones on the frame.
        """
        # Draw connections
        for p1_id, p2_id in self.CONNECTIONS:
            if p1_id < len(landmarks) and p2_id < len(landmarks):
                x1, y1 = landmarks[p1_id][1], landmarks[p1_id][2]
                x2, y2 = landmarks[p2_id][1], landmarks[p2_id][2]
                cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2, cv2.LINE_AA)

        # Draw key joint points
        for idx, cx, cy, _ in landmarks:
            # Highlight index fingertip (8) and middle fingertip (12)
            if idx in (8, 12):
                cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx, cy), 4, (0, 180, 255), -1, cv2.LINE_AA)
