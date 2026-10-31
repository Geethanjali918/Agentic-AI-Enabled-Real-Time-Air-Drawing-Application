"""
dl_models/train_dl.py
---------------------
Training pipeline for Deep Learning models:
  1. DeepGestureNet (Deep Landmark Neural Network)
  2. StrokeCNN (Deep Neural Network for Sketch & Shape Recognition)

Saves trained weights, convergence histories, and performance metrics.
"""

import os
import sys
import json
import time
import joblib
import cv2
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dl_models.deep_gesture_net import DeepGestureNet
from dl_models.stroke_cnn import StrokeCNN
from dataset.shapes_dataset import generate_shapes_dataset


def train_deep_gesture_net(dataset_csv="dataset/hand_gestures.csv", epochs=60, lr=0.001):
    print("\n" + "=" * 60)
    print("  TRAINING DEEP GESTURE NEURAL NETWORK (DNN)")
    print("=" * 60)

    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Dataset not found at {dataset_csv}")

    df = pd.read_csv(dataset_csv)
    feature_cols = [c for c in df.columns if c not in ["gesture", "timestamp", "sample_id"]]
    X = df[feature_cols].values.astype(np.float32)

    class_names = sorted(list(df["gesture"].unique()))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = np.array([class_to_idx[g] for g in df["gesture"]], dtype=np.int64)

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.80 * len(X))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"[DeepNet] Total Samples: {len(X)} | Input Dim: {X.shape[1]} | Classes: {class_names}")
    print(f"[DeepNet] Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    model = DeepGestureNet(input_dim=X.shape[1], hidden_dims=[128, 64, 32], num_classes=len(class_names))

    t0 = time.time()
    history = model.fit(X_train, y_train, X_val=X_val, y_val=y_val, epochs=epochs, batch_size=32, lr=lr)
    train_duration = time.time() - t0

    t_inf0 = time.time()
    val_preds, val_conf, _ = model.predict(X_val)
    val_latency_ms = ((time.time() - t_inf0) / len(X_val)) * 1000.0
    val_acc = float(np.mean(val_preds == y_val))

    print(f"\n[DeepNet Training Complete]")
    print(f"  Training Time: {train_duration:.2f}s")
    print(f"  Final Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"  Inference Latency: {val_latency_ms:.3f} ms / frame")

    weights_path = "dl_models/deep_gesture_weights.npz"
    metadata_path = "dl_models/dl_metadata.json"
    model.save_weights(weights_path)

    metadata = {
        "model_architecture": "DeepGestureNet (82 -> 128 -> 64 -> 32 -> 5)",
        "input_dimension": X.shape[1],
        "class_names": class_names,
        "epochs": epochs,
        "val_accuracy": round(val_acc, 4),
        "inference_latency_ms": round(val_latency_ms, 3),
        "history": history
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model, metadata


def train_stroke_cnn(shapes_dataset_path="dataset/air_shapes_dataset.npz"):
    print("\n" + "=" * 60)
    print("  TRAINING AIR-CANVAS SKETCH DEEP LEARNING MODEL")
    print("=" * 60)

    if not os.path.exists(shapes_dataset_path):
        print("[StrokeCNN] Shapes dataset not found. Generating now...")
        generate_shapes_dataset(shapes_dataset_path, samples_per_class=300)

    data = np.load(shapes_dataset_path)
    images = data["images"]  # shape (1500, 28, 28)
    labels = data["labels"]  # shape (1500,)
    class_names = [str(c) for c in data["class_names"]]

    print(f"[StrokeCNN] Loaded {len(images)} sketch samples across {len(class_names)} classes: {class_names}")

    X = images.reshape(len(images), -1)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    print(f"[StrokeCNN] Training Deep Multi-Layer Network (784 -> 128 -> 64 -> {len(class_names)})...")
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=250, random_state=42)
    mlp.fit(X_train, y_train)
    train_duration = time.time() - t0

    test_preds = mlp.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"\n[StrokeCNN Complete]")
    print(f"  Training Time: {train_duration:.2f}s")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")

    model_path = "dl_models/stroke_cnn_model.pkl"
    joblib.dump(mlp, model_path)
    print(f"[StrokeCNN] Model saved to {model_path}")
    return mlp


if __name__ == "__main__":
    train_deep_gesture_net()
    train_stroke_cnn()
