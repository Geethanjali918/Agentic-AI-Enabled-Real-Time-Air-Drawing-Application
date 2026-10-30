"""
gestures.py
-----------
Intuitive gesture recognition and pointer coordinate smoothing for Air Drawing.

Recognizes simple and natural hand poses:
- Index finger UP (others DOWN)       -> DRAW MODE
- Two fingers UP (Index + Middle)     -> SELECTION / CONTROL MODE
- Closed fist (all fingers folded)    -> ERASER MODE
- Index finger DOWN / Relaxed         -> STANDBY MODE
"""

import math


class GestureRecognizer:
    """
    Classifies hand poses into drawing, selection, eraser, or standby modes.
    """

    MODE_DRAW = "DRAW"
    MODE_SELECT = "SELECT"
    MODE_ERASE = "ERASE"
    MODE_STANDBY = "STANDBY"

    # Landmark indices for finger tips and PIP joints
    FINGER_TIPS = [8, 12, 16, 20]      # Index, Middle, Ring, Pinky
    FINGER_PIPS = [6, 10, 14, 18]

    def __init__(self):
        pass

    def get_finger_status(self, landmarks):
        """
        Determines whether each finger (Thumb, Index, Middle, Ring, Pinky) is extended (Up = 1, Down = 0).
        :param landmarks: list of (id, x, y, z)
        :return: list of 5 integers [thumb, index, middle, ring, pinky]
        """
        if not landmarks or len(landmarks) < 21:
            return [0, 0, 0, 0, 0]

        fingers = []

        # Thumb: compare x-distance relative to MCP or IP based on wrist position
        # For general upright hands, check thumb tip (4) distance from index MCP (5) vs wrist (0)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]

        # Calculate distance of thumb tip from pinky base (17) vs thumb MCP from pinky base
        pinky_mcp = landmarks[17]
        dist_tip_to_pinky = math.hypot(thumb_tip[1] - pinky_mcp[1], thumb_tip[2] - pinky_mcp[2])
        dist_mcp_to_pinky = math.hypot(thumb_mcp[1] - pinky_mcp[1], thumb_mcp[2] - pinky_mcp[2])
        thumb_up = 1 if dist_tip_to_pinky > dist_mcp_to_pinky * 1.1 else 0
        fingers.append(thumb_up)

        # 4 Fingers: Check if tip y is strictly above PIP joint y (in image coordinates, smaller y is higher)
        for tip_id, pip_id in zip(self.FINGER_TIPS, self.FINGER_PIPS):
            if landmarks[tip_id][2] < landmarks[pip_id][2]:
                fingers.append(1)  # Finger extended
            else:
                fingers.append(0)  # Finger curled

        return fingers

    def classify_gesture(self, landmarks):
        """
        Determines the current active mode based on finger states.
        :param landmarks: 21 landmark tuples
        :return: (mode_name, index_finger_point, middle_finger_point)
        """
        if not landmarks or len(landmarks) < 21:
            return self.MODE_STANDBY, None, None

        status = self.get_finger_status(landmarks)
        _, index_up, middle_up, ring_up, pinky_up = status

        index_pt = (landmarks[8][1], landmarks[8][2])
        middle_pt = (landmarks[12][1], landmarks[12][2])

        # 1. Closed fist (all 4 main fingers curled down) -> ERASER
        if not index_up and not middle_up and not ring_up and not pinky_up:
            # For fist, we can use the knuckle center (MCP 9 or wrist) as eraser center
            fist_center = (landmarks[9][1], landmarks[9][2])
            return self.MODE_ERASE, fist_center, None

        # 2. Only Index finger UP -> DRAW
        if index_up and not middle_up and not ring_up and not pinky_up:
            return self.MODE_DRAW, index_pt, None

        # 3. Index and Middle fingers UP -> SELECTION / CONTROL
        if index_up and middle_up and not ring_up and not pinky_up:
            # Return midpoint or index tip for selection pointer
            select_pt = ((index_pt[0] + middle_pt[0]) // 2, (index_pt[1] + middle_pt[1]) // 2)
            return self.MODE_SELECT, select_pt, middle_pt

        # 4. Any other configuration (open palm, hand relaxed, etc.) -> STANDBY
        return self.MODE_STANDBY, index_pt, None


class SmoothPointer:
    """
    Exponential Moving Average (EMA) filter to remove high-frequency hand tremors
    and prevent sudden line jumping when tracking is temporarily interrupted.
    """

    def __init__(self, alpha=0.65, max_jump_distance=100.0):
        """
        :param alpha: Smoothing factor (0.0 = very heavy smoothing, 1.0 = raw coordinates)
                      0.65 provides great balance between zero-latency and jitter-free lines.
        :param max_jump_distance: Maximum allowable jump distance in pixels between frames.
        """
        self.alpha = alpha
        self.max_jump_distance = max_jump_distance
        self.smoothed_x = None
        self.smoothed_y = None

    def update(self, current_pt):
        """
        Smooth the current point coordinates.
        :param current_pt: (x, y) tuple or None
        :return: smoothed (x, y) tuple or None
        """
        if current_pt is None:
            self.reset()
            return None

        curr_x, curr_y = current_pt

        if self.smoothed_x is None or self.smoothed_y is None:
            self.smoothed_x = float(curr_x)
            self.smoothed_y = float(curr_y)
            return int(self.smoothed_x), int(self.smoothed_y)

        # Distance check to avoid teleportation artifacts
        dist = math.hypot(curr_x - self.smoothed_x, curr_y - self.smoothed_y)
        if dist > self.max_jump_distance:
            # Teleportation occurred, reset smoothing to new position immediately
            self.smoothed_x = float(curr_x)
            self.smoothed_y = float(curr_y)
            return int(self.smoothed_x), int(self.smoothed_y)

        # EMA formula
        self.smoothed_x = self.alpha * curr_x + (1.0 - self.alpha) * self.smoothed_x
        self.smoothed_y = self.alpha * curr_y + (1.0 - self.alpha) * self.smoothed_y

        return int(round(self.smoothed_x)), int(round(self.smoothed_y))

    def reset(self):
        """Reset the internal smoothing state."""
        self.smoothed_x = None
        self.smoothed_y = None
