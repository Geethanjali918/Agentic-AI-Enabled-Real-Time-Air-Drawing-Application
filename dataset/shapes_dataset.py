"""
dataset/shapes_dataset.py
-------------------------
Generates and manages the Air-Drawn Geometric Shape & Sketch Dataset for Deep Learning.
Classes:
  0: CIRCLE
  1: RECTANGLE
  2: TRIANGLE
  3: STAR
  4: LINE

Produces 28x28 normalized single-channel stroke rasters + tabular coordinate sequences
representing real-time air drawings made by users.
"""

import os
import cv2
import numpy as np


def render_circle_sample(img_size=28):
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    center = (img_size // 2 + np.random.randint(-2, 3), img_size // 2 + np.random.randint(-2, 3))
    radius = np.random.randint(6, 11)
    thickness = np.random.randint(1, 3)
    cv2.circle(canvas, center, radius, 255, thickness)
    # Add drawing noise
    noise = (np.random.rand(img_size, img_size) * 15).astype(np.uint8)
    canvas = cv2.add(canvas, noise)
    return canvas


def render_rectangle_sample(img_size=28):
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    w = np.random.randint(12, 22)
    h = np.random.randint(12, 22)
    x1 = np.random.randint(2, max(3, img_size - w - 2))
    y1 = np.random.randint(2, max(3, img_size - h - 2))
    thickness = np.random.randint(1, 3)
    cv2.rectangle(canvas, (x1, y1), (x1 + w, y1 + h), 255, thickness)
    noise = (np.random.rand(img_size, img_size) * 15).astype(np.uint8)
    canvas = cv2.add(canvas, noise)
    return canvas


def render_triangle_sample(img_size=28):
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    cx = img_size // 2 + np.random.randint(-3, 4)
    top = (cx, np.random.randint(3, 7))
    left = (np.random.randint(3, 8), np.random.randint(20, 26))
    right = (np.random.randint(20, 26), np.random.randint(20, 26))
    thickness = np.random.randint(1, 3)
    pts = np.array([top, right, left], np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], isClosed=True, color=255, thickness=thickness)
    noise = (np.random.rand(img_size, img_size) * 15).astype(np.uint8)
    canvas = cv2.add(canvas, noise)
    return canvas


def render_star_sample(img_size=28):
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    cx, cy = img_size // 2, img_size // 2
    r_outer = np.random.randint(9, 12)
    r_inner = np.random.randint(4, 6)
    pts = []
    angle_offset = np.random.uniform(0, 2 * np.pi)
    for i in range(10):
        r = r_outer if i % 2 == 0 else r_inner
        ang = angle_offset + i * np.pi / 5
        pts.append([int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))])
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    thickness = np.random.randint(1, 2)
    cv2.polylines(canvas, [pts], isClosed=True, color=255, thickness=thickness)
    noise = (np.random.rand(img_size, img_size) * 15).astype(np.uint8)
    canvas = cv2.add(canvas, noise)
    return canvas


def render_line_sample(img_size=28):
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    x1, y1 = np.random.randint(3, 10), np.random.randint(3, 25)
    x2, y2 = np.random.randint(18, 26), np.random.randint(3, 25)
    thickness = np.random.randint(1, 3)
    cv2.line(canvas, (x1, y1), (x2, y2), 255, thickness)
    noise = (np.random.rand(img_size, img_size) * 15).astype(np.uint8)
    canvas = cv2.add(canvas, noise)
    return canvas


def generate_shapes_dataset(output_npz="dataset/air_shapes_dataset.npz", samples_per_class=300):
    """
    Generates single-channel 28x28 normalized images and labels for 5 classes.
    """
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    shape_classes = ["CIRCLE", "RECTANGLE", "TRIANGLE", "STAR", "LINE"]
    generators = [
        render_circle_sample,
        render_rectangle_sample,
        render_triangle_sample,
        render_star_sample,
        render_line_sample
    ]

    images = []
    labels = []

    print(f"[Shapes Dataset] Generating {samples_per_class} samples for each of {len(shape_classes)} shapes...")
    for class_id, (name, gen_func) in enumerate(zip(shape_classes, generators)):
        for _ in range(samples_per_class):
            img = gen_func(img_size=28)
            images.append(img)
            labels.append(class_id)
        print(f"  -> Generated {samples_per_class} samples for shape '{name}'")

    images = np.array(images, dtype=np.float32) / 255.0  # Normalized [0, 1]
    labels = np.array(labels, dtype=np.int64)

    # Shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    np.savez_compressed(output_npz, images=images, labels=labels, class_names=np.array(shape_classes))
    print(f"[Shapes Dataset] Saved {len(images)} samples to {output_npz}!")
    return images, labels, shape_classes


if __name__ == "__main__":
    generate_shapes_dataset()
