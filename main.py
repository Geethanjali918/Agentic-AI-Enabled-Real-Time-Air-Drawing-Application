"""
main.py
-------
Main entry point for the Real-Time Air Drawing Application.

Features:
- Real-time webcam tracking with MediaPipe
- Natural gesture recognition:
    * Index finger UP: Draw smooth lines
    * Two fingers UP (Index + Middle): Control / Button Selection Mode
    * Closed Fist: Circular Eraser
    * Index finger DOWN / Relaxed: Standby (no drawing)
- Intuitive Toolbar Interaction:
    * Touch buttons with 1 finger (Index) or 2 fingers (Select)
    * Click buttons directly with Mouse
    * Hotkeys: 1=Blue, 2=Green, 3=Red, 4=Black, C=Clear, S=Save
- Dual-buffer rendering (crisp drawing overlay including true black strokes)
- Error handling for camera connectivity and hand loss
"""

import sys
import time
import cv2
import numpy as np

from hand_tracking import HandTracker
from gestures import GestureRecognizer, SmoothPointer
from drawing import CanvasManager
from ai_interface import AICommandDispatcher


class AirDrawingApp:
    """
    Coordinates the camera feed, hand tracker, gesture recognizer,
    canvas manager, and user interface.
    """

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

        # Core system components
        self.tracker = HandTracker(max_num_hands=1)
        self.gesture_recognizer = GestureRecognizer()
        self.smooth_pointer = SmoothPointer(alpha=0.65, max_jump_distance=90.0)
        self.canvas = CanvasManager(width=640, height=480)
        self.ai = AICommandDispatcher(self.canvas)

        # Toolbar UI configuration
        self.toolbar_height = 70
        self.buttons = []
        self._init_toolbar()

        # Selection hover/dwell tracking
        self.hover_button_idx = None
        self.hover_start_time = 0.0
        self.dwell_time_required = 0.20  # Fast, responsive dwell time (0.2s)
        self.last_action_time = 0.0
        self.action_cooldown = 0.40       # Cooldown to prevent double clicks

        # Performance monitoring
        self.prev_frame_time = time.time()
        self.fps = 0.0

    def _init_toolbar(self):
        """Define on-screen interactive toolbar buttons."""
        # Button definitions: (Label, Action Type, Action Value, Background Color, Text Color)
        btn_defs = [
            ("1: BLUE", "color", "BLUE", (220, 80, 10), (255, 255, 255)),
            ("2: GREEN", "color", "GREEN", (30, 170, 40), (255, 255, 255)),
            ("3: RED", "color", "RED", (20, 30, 210), (255, 255, 255)),
            ("4: BLACK", "color", "BLACK", (30, 30, 30), (255, 255, 255)),
            ("ERASER", "tool", "ERASER", (70, 70, 70), (255, 255, 255)),
            ("CLEAR", "tool", "CLEAR", (40, 40, 140), (255, 255, 255)),
            ("SAVE", "tool", "SAVE", (120, 80, 20), (255, 255, 255)),
            ("SIZE", "tool", "SIZE", (60, 60, 60), (255, 255, 255)),
        ]

        total_buttons = len(btn_defs)
        margin_x = 8
        margin_y = 8
        spacing = 6
        available_width = 640 - (2 * margin_x) - ((total_buttons - 1) * spacing)
        btn_width = available_width // total_buttons
        btn_height = self.toolbar_height - (2 * margin_y)

        self.buttons = []
        for i, (label, action_type, action_val, bg_col, text_col) in enumerate(btn_defs):
            x1 = margin_x + i * (btn_width + spacing)
            y1 = margin_y
            x2 = x1 + btn_width
            y2 = y1 + btn_height
            self.buttons.append({
                "label": label,
                "type": action_type,
                "value": action_val,
                "rect": (x1, y1, x2, y2),
                "bg_color": bg_col,
                "text_color": text_col,
            })

    def open_camera(self):
        """Open webcam with graceful fallback to alternative indices and DirectShow on Windows."""
        print(f"[AirDrawing] Initializing webcam at index {self.camera_index}...")
        
        # Try DirectShow on Windows for near-instant camera startup
        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"[AirDrawing] Camera index {self.camera_index} failed. Trying index 1...")
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                if not self.cap or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(1)
            else:
                self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            print("\n" + "=" * 60)
            print("ERROR: Could not open any webcam.")
            print("Please check that:")
            print("  1. A webcam is connected to your computer.")
            print("  2. Camera permissions are enabled in Windows Settings.")
            print("  3. No other application is exclusively using the webcam.")
            print("=" * 60 + "\n")
            return False

        # Set resolution to 640x480 for smooth 30+ FPS tracking
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[AirDrawing] Webcam initialized successfully!")
        return True

    def _handle_toolbar_action(self, button):
        """Execute action associated with a toolbar button."""
        b_type = button["type"]
        val = button["value"]

        if b_type == "color":
            self.canvas.set_color(val)
        elif b_type == "tool":
            if val == "CLEAR":
                self.canvas.clear()
            elif val == "SAVE":
                self.canvas.save()
            elif val == "SIZE":
                self.canvas.cycle_brush_size()
            elif val == "ERASER":
                self.canvas.show_toast("Close fist to erase strokes")

    def _mouse_callback(self, event, x, y, flags, param):
        """Allows direct mouse clicking on toolbar buttons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in self.buttons:
                x1, y1, x2, y2 = btn["rect"]
                # Generous click hitbox for toolbar
                if (x1 - 5) <= x <= (x2 + 5) and 0 <= y <= (self.toolbar_height + 15):
                    self._handle_toolbar_action(btn)
                    break

    def draw_ui(self, frame, active_mode):
        """Draw the top toolbar, status badge, cursor, and bottom instructions."""
        h, w, _ = frame.shape

        # 1. Semi-transparent top toolbar bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, self.toolbar_height + 5), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Draw toolbar buttons
        now = time.time()
        for idx, btn in enumerate(self.buttons):
            x1, y1, x2, y2 = btn["rect"]
            is_hovered = (idx == self.hover_button_idx)
            is_active_color = (btn["type"] == "color" and btn["value"] == self.canvas.current_color_name)

            # Determine button background color
            bg = list(btn["bg_color"])
            if is_hovered:
                # Brighten button when hovered
                bg = [min(255, c + 45) for c in bg]

            cv2.rectangle(frame, (x1, y1), (x2, y2), tuple(bg), -1)

            # Active color gets a prominent glowing double border
            if is_active_color:
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 255), 3, cv2.LINE_AA)
                cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (255, 255, 255), 1, cv2.LINE_AA)
            elif is_hovered:
                # Draw dwell progress fill
                progress = min(1.0, (now - self.hover_start_time) / self.dwell_time_required)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                pw = int((x2 - x1) * progress)
                cv2.rectangle(frame, (x1, y2 - 5), (x1 + pw, y2), (0, 255, 0), -1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (75, 75, 75), 1, cv2.LINE_AA)

            # Button Text
            label = btn["label"]
            if btn["value"] == "SIZE":
                label = f"{self.canvas.brush_size}px"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.36
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            tx = x1 + ((x2 - x1) - tw) // 2
            ty = y1 + ((y2 - y1) + th) // 2
            cv2.putText(frame, label, (tx, ty), font, font_scale, btn["text_color"], thickness, cv2.LINE_AA)

        # 2. Mode Status Badge (Under toolbar)
        mode_colors = {
            GestureRecognizer.MODE_DRAW: (0, 220, 100),       # Bright Green
            GestureRecognizer.MODE_SELECT: (0, 220, 255),     # Yellow / Cyan
            GestureRecognizer.MODE_ERASE: (0, 70, 255),       # Orange / Red
            GestureRecognizer.MODE_STANDBY: (160, 160, 160),  # Gray
        }
        mode_col = mode_colors.get(active_mode, (200, 200, 200))
        status_text = f"MODE: {active_mode} | ACTIVE COLOR: {self.canvas.current_color_name} | BRUSH: {self.canvas.brush_size}px"
        cv2.putText(frame, status_text, (15, self.toolbar_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, mode_col, 2, cv2.LINE_AA)

        # 3. FPS display
        fps_text = f"FPS: {int(self.fps)}"
        cv2.putText(frame, fps_text, (w - 75, self.toolbar_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # 4. Bottom Instruction Bar
        overlay_bottom = frame.copy()
        cv2.rectangle(overlay_bottom, (0, h - 35), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay_bottom, 0.8, frame, 0.2, 0, frame)

        instructions = "☝️ Point at Buttons to select color  |  ✊ Fist: Erase  |  Keys: 1=Blue 2=Green 3=Red 4=Black [C]lear [S]ave [Q]uit"
        cv2.putText(frame, instructions, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (240, 240, 240), 1, cv2.LINE_AA)

        # 5. Toast notification
        self.canvas.draw_toast_overlay(frame)

    def draw_cursor(self, frame, pointer_pt, active_mode):
        """Render visual cursor at user's fingertip point."""
        if pointer_pt is None:
            return

        cx, cy = pointer_pt
        in_toolbar = (cy <= self.toolbar_height + 15)

        if in_toolbar:
            # When inside toolbar, show circular selection pointer
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        elif active_mode == GestureRecognizer.MODE_DRAW:
            # Circle showing current brush size and color
            cv2.circle(frame, (cx, cy), max(self.canvas.brush_size // 2, 4), self.canvas.current_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), max(self.canvas.brush_size // 2, 4) + 2, (255, 255, 255), 1, cv2.LINE_AA)
        elif active_mode == GestureRecognizer.MODE_SELECT:
            # Selection reticle
            cv2.circle(frame, (cx, cy), 9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
        elif active_mode == GestureRecognizer.MODE_ERASE:
            # Eraser circular boundary
            cv2.circle(frame, (cx, cy), self.canvas.eraser_radius, (0, 100, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (0, 100, 255), -1, cv2.LINE_AA)

    def process_selection_mode(self, select_pt):
        """Check if fingertip is touching or hovering over a toolbar button."""
        if select_pt is None:
            self.hover_button_idx = None
            return

        sx, sy = select_pt
        now = time.time()

        # Generous vertical and horizontal hit tolerance
        found_idx = None
        for idx, btn in enumerate(self.buttons):
            x1, y1, x2, y2 = btn["rect"]
            if (x1 - 5) <= sx <= (x2 + 5) and 0 <= sy <= (self.toolbar_height + 15):
                found_idx = idx
                break

        if found_idx is not None:
            if self.hover_button_idx == found_idx:
                # Continuous hover over same button
                if (now - self.hover_start_time) >= self.dwell_time_required and (now - self.last_action_time) > self.action_cooldown:
                    self._handle_toolbar_action(self.buttons[found_idx])
                    self.last_action_time = now
                    self.hover_start_time = now
            else:
                # Just moved into new button
                self.hover_button_idx = found_idx
                self.hover_start_time = now
        else:
            self.hover_button_idx = None

    def run(self):
        """Main application event loop."""
        if not self.open_camera():
            return

        window_name = "Air Drawing - College Project"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        # Register mouse click callback for instant mouse selection as well
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("\n" + "=" * 55)
        print(" Real-Time Air Drawing Application")
        print("=" * 55)
        print(" Controls:")
        print("   - Touch buttons at top with your finger or mouse to change color")
        print("   - 1 Finger UP (below toolbar): Draw")
        print("   - Closed Fist: Erase strokes")
        print("   - Keyboard Hotkeys:")
        print("       1 = Blue  |  2 = Green  |  3 = Red  |  4 = Black")
        print("       C = Clear |  S = Save   |  Q = Quit")
        print("=" * 55 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                # Flip horizontally for natural mirror behavior
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                self.canvas.resize(w, h)

                # 1. Hand tracking
                landmarks, frame = self.tracker.find_hand_landmarks(frame, draw=True)

                # 2. Gesture classification
                active_mode, raw_pointer, _ = self.gesture_recognizer.classify_gesture(landmarks)

                # 3. Pointer smoothing
                smoothed_pointer = self.smooth_pointer.update(raw_pointer)

                # 4. Handle gestures & toolbar interaction
                is_in_toolbar = (smoothed_pointer is not None and smoothed_pointer[1] <= self.toolbar_height + 15)

                if is_in_toolbar:
                    # Pointing inside toolbar zone triggers button selection (works with 1 or 2 fingers!)
                    self.canvas.reset_stroke()
                    self.process_selection_mode(smoothed_pointer)
                elif active_mode == GestureRecognizer.MODE_DRAW and smoothed_pointer is not None:
                    self.hover_button_idx = None
                    self.canvas.draw_stroke(smoothed_pointer)
                elif active_mode == GestureRecognizer.MODE_ERASE and smoothed_pointer is not None:
                    self.hover_button_idx = None
                    self.canvas.erase_at(smoothed_pointer)
                elif active_mode == GestureRecognizer.MODE_SELECT and smoothed_pointer is not None:
                    self.canvas.reset_stroke()
                    self.process_selection_mode(smoothed_pointer)
                else:
                    # STANDBY or no hand
                    self.canvas.reset_stroke()
                    self.hover_button_idx = None

                # 5. Blend canvas strokes onto webcam frame
                blended_frame = self.canvas.blend(frame)

                # 6. Render UI toolbar, status badge, instructions, and cursor
                self.draw_ui(blended_frame, active_mode)
                self.draw_cursor(blended_frame, smoothed_pointer, active_mode)

                # 7. Calculate FPS
                curr_time = time.time()
                time_diff = curr_time - self.prev_frame_time
                if time_diff > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
                self.prev_frame_time = curr_time

                # 8. Display output window
                cv2.imshow(window_name, blended_frame)

                # 9. Handle keyboard shortcuts (Hotkeys)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # 27 is ESC
                    print("[AirDrawing] Exiting application...")
                    break
                elif key in (ord('1'), ord('b'), ord('B')):
                    self.canvas.set_color("BLUE")
                elif key in (ord('2'), ord('g'), ord('G')):
                    self.canvas.set_color("GREEN")
                elif key in (ord('3'), ord('r'), ord('R')):
                    self.canvas.set_color("RED")
                elif key in (ord('4'), ord('k'), ord('K')):
                    self.canvas.set_color("BLACK")
                elif key in (ord('c'), ord('C')):
                    self.canvas.clear()
                elif key in (ord('s'), ord('S')):
                    self.canvas.save()
                elif key in (ord('e'), ord('E')):
                    self.canvas.show_toast("Close fist to erase strokes")
                elif key in (ord('+'), ord('=')):
                    self.canvas.cycle_brush_size()
                elif key in (ord('-'), ord('_')):
                    self.canvas.set_brush_size("SMALL")

        except KeyboardInterrupt:
            print("\n[AirDrawing] Interrupted by user.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release camera and destroy OpenCV windows safely."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("[AirDrawing] Resources successfully released. Goodbye!")


if __name__ == "__main__":
    app = AirDrawingApp()
    app.run()
