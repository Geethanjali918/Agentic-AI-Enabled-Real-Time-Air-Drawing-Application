"""
drawing.py
----------
Virtual canvas manager for smooth, real-time air drawing.
Uses a dual-buffer (Color Canvas + Stroke Mask) architecture so that Black brush
strokes display properly as solid black lines over the webcam feed rather than acting as an eraser.
"""

import os
import time
import cv2
import numpy as np


class CanvasManager:
    """
    Manages the virtual air canvas, drawing strokes, color palette, eraser,
    canvas clearing, and PNG exporting.
    """

    # Vibrant BGR colors
    COLORS = {
        "BLUE": (255, 110, 20),      # Vivid Royal Blue (BGR)
        "GREEN": (40, 210, 50),      # Fresh Emerald Green (BGR)
        "RED": (30, 40, 240),        # Vibrant Red (BGR)
        "BLACK": (25, 25, 25),       # Deep Crisp Black (BGR)
    }

    BRUSH_SIZES = {
        "SMALL": 4,
        "MEDIUM": 8,
        "LARGE": 16,
    }

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        # Dual-layer representation
        # canvas_color stores actual BGR colors drawn
        self.canvas_color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # canvas_mask is a 1-channel uint8 mask: 255 where strokes exist, 0 for empty space
        self.canvas_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Active drawing state
        self.current_color_name = "BLUE"
        self.current_color = self.COLORS[self.current_color_name]
        self.brush_size_name = "MEDIUM"
        self.brush_size = self.BRUSH_SIZES[self.brush_size_name]
        self.eraser_radius = 35

        # Tracking state to connect consecutive points
        self.prev_point = None

        # Toast / status notification message
        self.toast_message = ""
        self.toast_expiry = 0.0

    def resize(self, width, height):
        """Reinitialize canvas if camera resolution differs from default."""
        if self.width != width or self.height != height:
            self.width = width
            self.height = height
            self.canvas_color = np.zeros((height, width, 3), dtype=np.uint8)
            self.canvas_mask = np.zeros((height, width), dtype=np.uint8)
            self.prev_point = None

    def set_color(self, color_name):
        """Set the active drawing color by name ('BLUE', 'GREEN', 'RED', 'BLACK')."""
        color_name = color_name.upper()
        if color_name in self.COLORS:
            self.current_color_name = color_name
            self.current_color = self.COLORS[color_name]
            self.show_toast(f"Color: {color_name}")

    def cycle_brush_size(self):
        """Cycle through Small -> Medium -> Large brush sizes."""
        sizes = list(self.BRUSH_SIZES.keys())
        idx = (sizes.index(self.brush_size_name) + 1) % len(sizes)
        self.brush_size_name = sizes[idx]
        self.brush_size = self.BRUSH_SIZES[self.brush_size_name]
        self.show_toast(f"Brush Size: {self.brush_size_name} ({self.brush_size}px)")

    def set_brush_size(self, size_name_or_int):
        """Set brush size either by name ('SMALL', 'MEDIUM', 'LARGE') or exact pixel integer."""
        if isinstance(size_name_or_int, str) and size_name_or_int.upper() in self.BRUSH_SIZES:
            self.brush_size_name = size_name_or_int.upper()
            self.brush_size = self.BRUSH_SIZES[self.brush_size_name]
        elif isinstance(size_name_or_int, int) and 2 <= size_name_or_int <= 50:
            self.brush_size = size_name_or_int
            self.brush_size_name = f"{size_name_or_int}px"
        self.show_toast(f"Brush: {self.brush_size}px")

    def draw_stroke(self, current_point):
        """
        Draw a smooth stroke segment from prev_point to current_point.
        """
        if current_point is None:
            self.prev_point = None
            return

        cx, cy = current_point

        if self.prev_point is None:
            # First point of stroke - draw a rounded dot
            cv2.circle(self.canvas_color, (cx, cy), self.brush_size // 2, self.current_color, -1, cv2.LINE_AA)
            cv2.circle(self.canvas_mask, (cx, cy), self.brush_size // 2, 255, -1, cv2.LINE_AA)
            self.prev_point = (cx, cy)
            return

        px, py = self.prev_point
        # Draw smooth line connecting previous and current points
        cv2.line(self.canvas_color, (px, py), (cx, cy), self.current_color, self.brush_size, cv2.LINE_AA)
        cv2.line(self.canvas_mask, (px, py), (cx, cy), 255, self.brush_size, cv2.LINE_AA)

        # Add rounded caps at joint for seamless curves
        cv2.circle(self.canvas_color, (cx, cy), self.brush_size // 2, self.current_color, -1, cv2.LINE_AA)
        cv2.circle(self.canvas_mask, (cx, cy), self.brush_size // 2, 255, -1, cv2.LINE_AA)

        self.prev_point = (cx, cy)

    def erase_at(self, center_point, radius=None):
        """
        Erase strokes in a circular area around center_point.
        """
        if center_point is None:
            self.prev_point = None
            return

        r = radius if radius is not None else self.eraser_radius
        cx, cy = center_point

        # Clear mask and color in circle
        cv2.circle(self.canvas_mask, (cx, cy), r, 0, -1)
        cv2.circle(self.canvas_color, (cx, cy), r, (0, 0, 0), -1)

        # In erase mode, line connecting is not needed
        self.prev_point = None

    def clear(self):
        """Clear the entire canvas."""
        self.canvas_color.fill(0)
        self.canvas_mask.fill(0)
        self.prev_point = None
        self.show_toast("Canvas Cleared")

    def save(self, save_dir="saved_drawings"):
        """
        Export drawing as high-resolution PNG on clean white canvas.
        :param save_dir: Directory to save exported drawings
        :return: Path to saved file
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"air_drawing_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)

        # Create clean white background
        clean_canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        # Place strokes where mask is positive
        clean_canvas[self.canvas_mask > 0] = self.canvas_color[self.canvas_mask > 0]

        cv2.imwrite(filepath, clean_canvas)
        self.show_toast(f"Saved: {filename}", duration=3.0)
        return filepath

    def reset_stroke(self):
        """Break stroke continuity so next draw action does not connect to old position."""
        self.prev_point = None

    def blend(self, frame):
        """
        Blend virtual drawing strokes directly onto webcam frame.
        :param frame: Live webcam BGR frame
        :return: Blended BGR frame
        """
        output = frame.copy()
        # Direct crisp overlay where mask is positive
        stroke_indices = self.canvas_mask > 0
        output[stroke_indices] = self.canvas_color[stroke_indices]
        return output

    def show_toast(self, message, duration=2.0):
        """Display an on-screen toast notification for a specified duration."""
        self.toast_message = message
        self.toast_expiry = time.time() + duration

    def draw_toast_overlay(self, frame):
        """Draw active toast message on the frame if active."""
        if time.time() < self.toast_expiry and self.toast_message:
            text = self.toast_message
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.65
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

            # Center toast horizontally near bottom above instruction bar
            tx = (self.width - text_w) // 2
            ty = self.height - 55

            pad = 10
            # Draw dark rounded background badge
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (tx - pad, ty - text_h - pad),
                (tx + text_w + pad, ty + baseline + pad),
                (30, 30, 30),
                -1
            )
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(
                frame,
                (tx - pad, ty - text_h - pad),
                (tx + text_w + pad, ty + baseline + pad),
                (0, 200, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(frame, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
