"""
ml_models/ml_gesture_classifier.py
----------------------------------
Real-time Machine Learning Gesture Classifier for Air Drawing.
Loads the trained Champion Model (Random Forest or SVM) and provides instant,
low-latency predictions (under 1ms) for each webcam frame.
"""

import os
import json
import joblib
import numpy as np

from dataset.feature_extractor import HandFeatureExtractor


# Ensure absolute paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MLGestureClassifier:
    """
    Inference engine that transforms raw MediaPipe landmarks into
    gesture predictions using the trained Machine Learning model.
    """

    def __init__(self, model_path=None, encoder_path=None, metadata_path=None):
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "ml_models", "best_gesture_model.pkl")
        if encoder_path is None:
            encoder_path = os.path.join(PROJECT_ROOT, "ml_models", "label_encoder.pkl")
        if metadata_path is None:
            metadata_path = os.path.join(PROJECT_ROOT, "ml_models", "model_metadata.json")

        self.extractor = HandFeatureExtractor()
        self.model = None
        self.encoder = None
        self.class_names = ["DRAW", "SELECT", "ERASE", "STANDBY", "CLEAR"]
        self.model_name = "Heuristic (Fallback)"
        self.is_loaded = False

        if os.path.exists(model_path) and os.path.exists(encoder_path):
            try:
                self.model = joblib.load(model_path)
                self.encoder = joblib.load(encoder_path)
                self.class_names = list(self.encoder.classes_)
                self.is_loaded = True

                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        meta = json.load(f)
                        self.model_name = meta.get("best_model_name", "Machine Learning Model")
                else:
                    self.model_name = type(self.model).__name__

                print(f"[ML Classifier] Loaded '{self.model_name}' successfully with {len(self.class_names)} classes.")
            except Exception as e:
                print(f"[ML Classifier Warning] Failed to load ML model: {e}. Using fallback.")
                self.is_loaded = False
        else:
            print("[ML Classifier] Model files not found. Run 'train_ml.py' to generate trained weights.")

    def predict(self, landmarks):
        """
        Predict gesture class and confidence from 21 MediaPipe landmarks.
        :param landmarks: list of 21 landmark tuples (id, x, y, z)
        :return: (predicted_gesture_name, confidence_float, proba_dict)
        """
        if not self.is_loaded or self.model is None or not landmarks or len(landmarks) < 21:
            return "STANDBY", 0.0, {}

        features = self.extractor.extract_features(landmarks)
        if features is None:
            return "STANDBY", 0.0, {}

        X = features.reshape(1, -1)

        # Get probabilities if supported
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            proba_dict = {name: float(p) for name, p in zip(self.class_names, probs)}
        else:
            pred_idx = int(self.model.predict(X)[0])
            conf = 1.0
            proba_dict = {self.class_names[pred_idx]: 1.0}

        pred_name = self.class_names[pred_idx]
        return pred_name, conf, proba_dict
