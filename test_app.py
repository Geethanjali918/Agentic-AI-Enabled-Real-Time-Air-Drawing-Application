"""
test_app.py
-----------
Comprehensive automated test suite to verify:
1. Hand tracking initialization and inference on synthetic frames
2. Gesture classification logic (Draw, Select, Erase, Standby)
3. Smoothing pointer algorithms
4. Virtual canvas drawing, erasing, clearing, blending, and PNG export
5. AI Command Dispatcher integration
"""

import os
import cv2
import numpy as np

from hand_tracking import HandTracker
from gestures import GestureRecognizer, SmoothPointer
from drawing import CanvasManager
from ai_interface import AICommandDispatcher


def test_hand_tracker():
    print("[TEST 1/5] Testing HandTracker initialization and inference...")
    tracker = HandTracker(max_num_hands=1)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a mock white circle in dummy frame
    cv2.circle(dummy_frame, (320, 240), 50, (255, 255, 255), -1)

    landmarks, processed = tracker.find_hand_landmarks(dummy_frame, draw=True)
    assert processed.shape == (480, 640, 3), "Processed frame shape mismatch"
    assert isinstance(landmarks, list), "Landmarks should be a list"
    print("  -> HandTracker initialized and processed frame successfully!")


def test_gestures():
    print("[TEST 2/5] Testing GestureRecognizer on synthetic hand landmarks...")
    recognizer = GestureRecognizer()

    # Helper to generate synthetic upright landmarks
    # (id, x, y, z)
    def make_landmarks(index_up=False, middle_up=False, ring_up=False, pinky_up=False):
        lms = [(i, 300, 300, 0.0) for i in range(21)]
        lms[0] = (0, 300, 400, 0.0)    # Wrist
        lms[17] = (17, 350, 320, 0.0)  # Pinky base
        lms[2] = (2, 250, 360, 0.0)    # Thumb MCP
        lms[4] = (4, 230, 350, 0.0)    # Thumb tip

        # Index: PIP 6, TIP 8
        lms[6] = (6, 280, 250, 0.0)
        lms[8] = (8, 280, 180 if index_up else 280, 0.0)

        # Middle: PIP 10, TIP 12
        lms[10] = (10, 300, 240, 0.0)
        lms[12] = (12, 300, 170 if middle_up else 270, 0.0)

        # Ring: PIP 14, TIP 16
        lms[14] = (14, 320, 250, 0.0)
        lms[16] = (16, 320, 180 if ring_up else 280, 0.0)

        # Pinky: PIP 18, TIP 20
        lms[18] = (18, 340, 260, 0.0)
        lms[20] = (20, 340, 190 if pinky_up else 290, 0.0)

        return lms

    # 1. DRAW: only index up
    draw_lms = make_landmarks(index_up=True, middle_up=False, ring_up=False, pinky_up=False)
    mode, pt, _ = recognizer.classify_gesture(draw_lms)
    assert mode == GestureRecognizer.MODE_DRAW, f"Expected DRAW, got {mode}"
    assert pt == (280, 180), f"Expected tip coords (280, 180), got {pt}"
    print("  -> Draw mode correctly recognized!")

    # 2. SELECT: index + middle up
    select_lms = make_landmarks(index_up=True, middle_up=True, ring_up=False, pinky_up=False)
    mode, pt, _ = recognizer.classify_gesture(select_lms)
    assert mode == GestureRecognizer.MODE_SELECT, f"Expected SELECT, got {mode}"
    print("  -> Select mode correctly recognized!")

    # 3. ERASE: closed fist (all 4 down)
    fist_lms = make_landmarks(index_up=False, middle_up=False, ring_up=False, pinky_up=False)
    mode, pt, _ = recognizer.classify_gesture(fist_lms)
    assert mode == GestureRecognizer.MODE_ERASE, f"Expected ERASE, got {mode}"
    print("  -> Erase mode correctly recognized!")

    # 4. STANDBY: all up
    standby_lms = make_landmarks(index_up=True, middle_up=True, ring_up=True, pinky_up=True)
    mode, pt, _ = recognizer.classify_gesture(standby_lms)
    assert mode == GestureRecognizer.MODE_STANDBY, f"Expected STANDBY, got {mode}"
    print("  -> Standby mode correctly recognized!")


def test_smoothing():
    print("[TEST 3/5] Testing SmoothPointer filter...")
    sp = SmoothPointer(alpha=0.5, max_jump_distance=80.0)

    # Initial point
    p1 = sp.update((100, 100))
    assert p1 == (100, 100), f"Initial point should be (100, 100), got {p1}"

    # Smooth step
    p2 = sp.update((120, 120))
    # 0.5 * 120 + 0.5 * 100 = 110
    assert p2 == (110, 110), f"Smoothed point should be (110, 110), got {p2}"

    # Sudden jump > 80 pixels should reset rather than stretch
    p3 = sp.update((300, 300))
    assert p3 == (300, 300), f"Teleport jump should reset to (300, 300), got {p3}"
    print("  -> SmoothPointer passed smoothing and jump-protection tests!")


def test_drawing_and_canvas():
    print("[TEST 4/5] Testing CanvasManager operations...")
    canvas = CanvasManager(width=320, height=240)

    # Colors
    canvas.set_color("RED")
    assert canvas.current_color_name == "RED"
    assert canvas.current_color == (30, 40, 240)

    canvas.set_color("BLACK")
    assert canvas.current_color_name == "BLACK"

    # Draw strokes
    canvas.draw_stroke((50, 50))
    canvas.draw_stroke((60, 60))
    canvas.draw_stroke((70, 70))
    assert np.count_nonzero(canvas.canvas_mask) > 0, "Stroke should be recorded on mask"

    # Erase
    strokes_before_erase = np.count_nonzero(canvas.canvas_mask)
    canvas.erase_at((60, 60), radius=20)
    strokes_after_erase = np.count_nonzero(canvas.canvas_mask)
    assert strokes_after_erase < strokes_before_erase, "Eraser should remove pixels"

    # Save to disk
    test_dir = "test_saved_drawings"
    saved_file = canvas.save(save_dir=test_dir)
    assert os.path.exists(saved_file), f"Saved file should exist at {saved_file}"
    img = cv2.imread(saved_file)
    assert img.shape == (240, 320, 3), f"Saved image shape mismatch: {img.shape}"
    print(f"  -> Successfully exported clean PNG to {saved_file}")

    # Clear
    canvas.clear()
    assert np.count_nonzero(canvas.canvas_mask) == 0, "Canvas mask should be completely empty after clear"
    assert np.count_nonzero(canvas.canvas_color) == 0, "Canvas color should be empty after clear"

    # Cleanup test artifact
    if os.path.exists(saved_file):
        os.remove(saved_file)
    if os.path.exists(test_dir):
        os.rmdir(test_dir)
    print("  -> CanvasManager passed all drawing, erasing, clearing, and saving tests!")


def test_ai_interface():
    print("[TEST 5/5] Testing AICommandDispatcher...")
    canvas = CanvasManager(width=320, height=240)
    ai = AICommandDispatcher(canvas)

    # Test "Change color to blue"
    res = ai.execute_command("Change the color to blue")
    assert res["status"] == "success"
    assert canvas.current_color_name == "BLUE"

    # Test "Clear canvas"
    res = ai.execute_command("Please clear the canvas now")
    assert res["status"] == "success"
    assert res["action"] == "clear"

    # Test "Increase brush size"
    res = ai.execute_command("Make brush bigger")
    assert res["status"] == "success"
    assert canvas.brush_size_name == "LARGE"

    # Test "Save drawing"
    res = ai.execute_command("Save the drawing")
    assert res["status"] == "success"
    assert os.path.exists(res["filepath"])
    os.remove(res["filepath"])
    if os.path.exists("saved_drawings") and len(os.listdir("saved_drawings")) == 0:
        os.rmdir("saved_drawings")

    print("  -> AICommandDispatcher passed all natural language command tests!")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING AIR DRAWING AUTOMATED TEST SUITE")
    print("=" * 60)
    test_hand_tracker()
    test_gestures()
    test_smoothing()
    test_drawing_and_canvas()
    test_ai_interface()
    print("=" * 60)
    print(" ALL 5/5 TEST SUITES PASSED PERFECTLY!")
    print("=" * 60)
