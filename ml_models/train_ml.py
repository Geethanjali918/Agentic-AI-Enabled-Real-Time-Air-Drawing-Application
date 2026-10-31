"""
ml_models/train_ml.py
---------------------
Trains and benchmarks Machine Learning models on the Hand Gesture Dataset.
Models trained & compared:
  1. Random Forest Classifier (RFC)
  2. Support Vector Machine (SVM, RBF Kernel)
  3. K-Nearest Neighbors (KNN)
  4. Logistic Regression (Baseline)

Evaluates:
  - 5-Fold Stratified Cross-Validation Accuracy
  - Test Set Accuracy, Precision, Recall, F1-score
  - Per-class Confusion Matrices
  - Inference Latency (ms per prediction)

Saves the top-performing model to ml_models/best_gesture_model.pkl
along with comprehensive metadata to ml_models/model_metadata.json.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Ensure root directory is on python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset_generator import create_full_dataset


def train_and_evaluate_models(dataset_path="dataset/hand_gestures.csv", model_dir="ml_models"):
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load dataset
    if not os.path.exists(dataset_path):
        print(f"[ML Training] Dataset not found at {dataset_path}. Generating dataset now...")
        df = create_full_dataset(output_csv=dataset_path, samples_per_class=500)
    else:
        print(f"[ML Training] Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)

    print(f"[ML Training] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns.")

    # 2. Separate features X and target y
    feature_cols = [c for c in df.columns if c not in ["gesture", "timestamp", "sample_id"]]
    X = df[feature_cols].values
    y_raw = df["gesture"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)
    print(f"[ML Training] Target classes ({len(class_names)}): {class_names}")

    # 3. Train-test split (80% train, 20% test stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[ML Training] Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 4. Define ML models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        "Support Vector Machine": SVC(C=10.0, kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}
    best_f1 = -1.0
    best_model_name = None
    best_model = None

    print("\n" + "=" * 65)
    print(f"{'Model':<25} | {'5-Fold CV Acc':<14} | {'Test Acc':<10} | {'Test F1':<10} | {'Latency':<8}")
    print("=" * 65)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        # Cross-validation
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        mean_cv_acc = float(np.mean(cv_scores))

        # Fit on full training set
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        # Predict test set
        t_infer_start = time.time()
        y_pred = clf.predict(X_test)
        t_infer_total = time.time() - t_infer_start
        latency_ms = (t_infer_total / len(X_test)) * 1000.0

        test_acc = float(accuracy_score(y_test, y_pred))
        test_f1 = float(f1_score(y_test, y_pred, average="weighted"))
        conf_mat = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        results[name] = {
            "cv_accuracy_mean": round(mean_cv_acc, 4),
            "cv_accuracy_std": round(float(np.std(cv_scores)), 4),
            "test_accuracy": round(test_acc, 4),
            "test_f1": round(test_f1, 4),
            "latency_ms": round(latency_ms, 3),
            "confusion_matrix": conf_mat,
            "classification_report": report
        }

        print(f"{name:<25} | {mean_cv_acc*100:>12.2f}% | {test_acc*100:>8.2f}% | {test_f1*100:>8.2f}% | {latency_ms:>6.2f}ms")

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_name = name
            best_model = clf

    print("=" * 65)
    print(f"\n[ML Training] Champion Model: '{best_model_name}' (Test F1: {best_f1*100:.2f}%)")

    # 5. Save champion model and label encoder
    model_save_path = os.path.join(model_dir, "best_gesture_model.pkl")
    encoder_save_path = os.path.join(model_dir, "label_encoder.pkl")
    metadata_save_path = os.path.join(model_dir, "model_metadata.json")

    joblib.dump(best_model, model_save_path)
    joblib.dump(label_encoder, encoder_save_path)

    metadata = {
        "best_model_name": best_model_name,
        "feature_names": feature_cols,
        "class_names": class_names,
        "n_features": len(feature_cols),
        "results": results
    }

    with open(metadata_save_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[ML Training] Model saved to {model_save_path}")
    print(f"[ML Training] Metadata saved to {metadata_save_path}")
    return results


if __name__ == "__main__":
    train_and_evaluate_models()
