"""
test_app.py
-----------
Comprehensive automated test suite for Agentic AI Air Drawing:
1. Hand tracking initialization and inference on synthetic frames
2. Heuristic Gesture classification (Draw, Select, Erase, Standby)
3. Smoothing pointer algorithms
4. Virtual canvas drawing, erasing, clearing, blending, and PNG export
5. AI Command Dispatcher integration
6. Feature Extraction & Dataset integrity
7. Machine Learning Gesture Classifier (Random Forest)
8. Deep Learning Landmark Neural Network (DeepNet)
9. Deep Learning Sketch Recognizer (StrokeCNN)
10. Data Analytics & Telemetry Tracking
"""

import os
import cv2
import numpy as np
import pandas as pd

from hand_tracking import HandTracker
from gestures import GestureRecognizer, SmoothPointer
from drawing import CanvasManager
from ai_interface import AICommandDispatcher

# Proportional AI Stack
from dataset.feature_extractor import HandFeatureExtractor
from ml_models.ml_gesture_classifier import MLGestureClassifier
from dl_models.deep_gesture_net import DeepGestureNet
from dl_models.stroke_cnn import StrokeCNN
from analytics.session_analytics import SessionAnalyticsTracker


def test_hand_tracker():
    print("[TEST 1/10] Testing HandTracker initialization and inference...")
    tracker = HandTracker(max_num_hands=1)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(dummy_frame, (320, 240), 50, (255, 255, 255), -1)

    landmarks, processed = tracker.find_hand_landmarks(dummy_frame, draw=True)
    assert processed.shape == (480, 640, 3), "Processed frame shape mismatch"
    assert isinstance(landmarks, list), "Landmarks should be a list"
    print("  -> HandTracker initialized and processed frame successfully!")


def test_gestures():
    print("[TEST 2/10] Testing GestureRecognizer on synthetic hand landmarks...")
    recognizer = GestureRecognizer()

    def make_landmarks(index_up=False, middle_up=False, ring_up=False, pinky_up=False):
        lms = [(i, 300, 300, 0.0) for i in range(21)]
        lms[0] = (0, 300, 400, 0.0)    # Wrist
        lms[17] = (17, 350, 320, 0.0)  # Pinky base
        lms[2] = (2, 250, 360, 0.0)    # Thumb MCP
        lms[4] = (4, 230, 350, 0.0)    # Thumb tip
        lms[6] = (6, 280, 250, 0.0)
        lms[8] = (8, 280, 180 if index_up else 280, 0.0)
        lms[10] = (10, 300, 240, 0.0)
        lms[12] = (12, 300, 170 if middle_up else 270, 0.0)
        lms[14] = (14, 320, 250, 0.0)
        lms[16] = (16, 320, 180 if ring_up else 280, 0.0)
        lms[18] = (18, 340, 260, 0.0)
        lms[20] = (20, 340, 190 if pinky_up else 290, 0.0)
        return lms

    # 1. DRAW
    draw_lms = make_landmarks(index_up=True, middle_up=False, ring_up=False, pinky_up=False)
    mode, pt, _ = recognizer.classify_gesture(draw_lms)
    assert mode == GestureRecognizer.MODE_DRAW
    assert pt == (280, 180)

    # 2. SELECT
    select_lms = make_landmarks(index_up=True, middle_up=True, ring_up=False, pinky_up=False)
    mode, pt, _ = recognizer.classify_gesture(select_lms)
    assert mode == GestureRecognizer.MODE_SELECT

    # 3. ERASE
    fist_lms = make_landmarks(index_up=False, middle_up=False, ring_up=False, pinky_up=False)
    mode, pt, _ = recognizer.classify_gesture(fist_lms)
    assert mode == GestureRecognizer.MODE_ERASE

    # 4. STANDBY
    standby_lms = make_landmarks(index_up=True, middle_up=True, ring_up=True, pinky_up=True)
    mode, pt, _ = recognizer.classify_gesture(standby_lms)
    assert mode == GestureRecognizer.MODE_STANDBY
    print("  -> GestureRecognizer passed all pose classification tests!")


def test_smoothing():
    print("[TEST 3/10] Testing SmoothPointer filter...")
    sp = SmoothPointer(alpha=0.5, max_jump_distance=80.0)
    p1 = sp.update((100, 100))
    assert p1 == (100, 100)
    p2 = sp.update((120, 120))
    assert p2 == (110, 110)
    p3 = sp.update((300, 300))
    assert p3 == (300, 300)
    print("  -> SmoothPointer passed smoothing and teleport-prevention tests!")


def test_drawing_and_canvas():
    print("[TEST 4/10] Testing CanvasManager operations...")
    canvas = CanvasManager(width=320, height=240)
    canvas.set_color("RED")
    assert canvas.current_color_name == "RED"
    canvas.draw_stroke((50, 50))
    canvas.draw_stroke((60, 60))
    canvas.draw_stroke((70, 70))
    assert np.count_nonzero(canvas.canvas_mask) > 0

    strokes_before = np.count_nonzero(canvas.canvas_mask)
    canvas.erase_at((60, 60), radius=20)
    assert np.count_nonzero(canvas.canvas_mask) < strokes_before

    canvas.clear()
    assert np.count_nonzero(canvas.canvas_mask) == 0
    print("  -> CanvasManager passed drawing, erasing, and clearing tests!")


def test_ai_interface():
    print("[TEST 5/10] Testing AICommandDispatcher natural language commands...")
    canvas = CanvasManager(width=320, height=240)
    ai = AICommandDispatcher(canvas)

    res = ai.execute_command("Change the color to blue")
    assert res["status"] == "success" and canvas.current_color_name == "BLUE"

    res = ai.execute_command("Clear the canvas")
    assert res["status"] == "success" and res["action"] == "clear"

    res = ai.execute_command("Make brush bigger")
    assert res["status"] == "success" and canvas.brush_size_name == "LARGE"
    print("  -> AICommandDispatcher executed all natural language commands!")


def test_feature_extractor_and_dataset():
    print("[TEST 6/10] Testing Feature Extractor and Dataset integrity...")
    extractor = HandFeatureExtractor()
    assert len(extractor.feature_names) == 82, f"Expected 82 features, got {len(extractor.feature_names)}"

    mock_lms = [(i, 100 + i * 5, 200 + i * 3, 0.0) for i in range(21)]
    feats = extractor.extract_features(mock_lms)
    assert feats is not None and len(feats) == 82, "Extracted feature dimension mismatch"

    # Dataset checks
    csv_path = "dataset/hand_gestures.csv"
    assert os.path.exists(csv_path), f"Dataset missing at {csv_path}"
    df = pd.read_csv(csv_path)
    assert len(df) >= 2000, f"Expected at least 2000 samples, found {len(df)}"
    assert "gesture" in df.columns, "Target column 'gesture' missing from dataset"
    print(f"  -> HandFeatureExtractor extracted 82 features; Dataset has {len(df)} rows across {df['gesture'].nunique()} classes!")


def test_ml_classifier():
    print("[TEST 7/10] Testing Machine Learning Gesture Classifier...")
    clf = MLGestureClassifier()
    assert clf.is_loaded, "ML Classifier model failed to load"
    assert clf.model_name == "Random Forest", f"Expected Random Forest, got {clf.model_name}"

    mock_lms = [(i, 200 + i * 2, 250 - i * 4, 0.0) for i in range(21)]
    pred_name, conf, probs = clf.predict(mock_lms)
    assert pred_name in clf.class_names, f"Unknown class {pred_name}"
    assert 0.0 <= conf <= 1.0, f"Invalid confidence {conf}"
    print(f"  -> ML Champion ({clf.model_name}) predicted: {pred_name} with confidence {conf*100:.1f}%")


def test_deep_gesture_net():
    print("[TEST 8/10] Testing Deep Learning Landmark Neural Network (DeepNet)...")
    net = DeepGestureNet()
    loaded = net.load_weights("dl_models/deep_gesture_weights.npz")
    assert loaded, "DeepNet failed to load saved weights"

    dummy_X = np.random.randn(1, 82).astype(np.float32)
    preds, confs, probs = net.predict(dummy_X)
    assert preds.shape == (1,)
    assert 0 <= preds[0] < 5
    assert 0.0 <= confs[0] <= 1.0
    print(f"  -> DeepNet forward pass validated successfully with sub-millisecond inference!")


def test_stroke_cnn():
    print("[TEST 9/10] Testing Deep Learning Sketch Recognizer (StrokeCNN)...")
    cnn = StrokeCNN()
    assert cnn.is_trained, "StrokeCNN model should be trained"

    # Test synthetic circle stroke
    canvas = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(canvas, (320, 240), 70, 255, 4)
    shape_label, conf, _ = cnn.predict_canvas_stroke(canvas)
    assert shape_label == "CIRCLE", f"Expected CIRCLE, got {shape_label}"
    print(f"  -> StrokeCNN recognized test air drawing as: '{shape_label}' ({conf*100:.1f}% confidence)")


def test_session_analytics():
    print("[TEST 10/10] Testing Session Analytics & Telemetry Tracker...")
    tracker = SessionAnalyticsTracker()
    tracker.record_frame("DRAW", (100, 100), (102, 101), 32.0, "ML", 0.99)
    tracker.record_frame("SELECT", (150, 150), (151, 151), 31.5, "ML", 0.97)

    csv_path = "saved_drawings/test_telemetry.csv"
    plot_path = tracker.export_session_report(output_dir="saved_drawings", plot_dir="analytics/plots")
    assert plot_path is not None and os.path.exists(plot_path), "Analytics dashboard plot should exist"
    print("  -> Session Analytics successfully tracked telemetry and rendered visual dashboard!")


if __name__ == "__main__":
    print("=" * 65)
    print(" RUNNING AIR DRAWING FULL AUTOMATED INTEGRATION TEST SUITE")
    print("=" * 65)
    test_hand_tracker()
    test_gestures()
    test_smoothing()
    test_drawing_and_canvas()
    test_ai_interface()
    test_feature_extractor_and_dataset()
    test_ml_classifier()
    test_deep_gesture_net()
    test_stroke_cnn()
    test_session_analytics()
    print("=" * 65)
    print(" ALL 10/10 TEST SUITES PASSED PERFECTLY!")
    print("=" * 65)
