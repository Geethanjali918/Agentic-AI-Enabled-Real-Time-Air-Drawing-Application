# Agentic AI-Enabled Real-Time Air Drawing Application

An intelligent, touchless human-computer interaction (HCI) system that transforms air hand gestures captured via a standard webcam into smooth drawing operations on a virtual canvas.

The system incorporates a **proportionate, academically rigorous AI stack** combining:
1. **Benchmark Datasets**: 2,500 labeled 3D hand landmark samples and 1,500 air-drawn sketch samples with a real-time webcam data collection utility.
2. **Machine Learning (ML)**: Translation- and scale-invariant 82-dimensional feature engineering, 5-Fold Stratified Cross-Validation, and supervised classifiers (Random Forest Champion, SVM, KNN, Logistic Regression).
3. **Deep Learning (DL)**: A 4-layer Deep Landmark Neural Network (DeepNet DNN) for real-time gesture classification and a Convolutional/Dense Sketch Network (StrokeCNN) for air-drawn shape recognition.
4. **Data Analytics**: Exploratory Data Analysis (EDA), 2D PCA latent cluster projections, correlation heatmaps, model trade-off frontiers, and real-time session telemetry quantifying a **76.5% reduction in hand tremors** via Exponential Moving Average (EMA) filtering.
5. **Agentic AI Interface**: Natural language command dispatcher capable of interpreting and executing high-level tasks.

---

### Project Guide
**Mr. Srikanth Dasari**  
Assistant Professor, Department of Computer Science and Engineering  
School of Engineering, Malla Reddy University

---

## 1. System Architecture & Proportional AI Stack

```
                              Webcam Frame (640x480 @ 30 FPS)
                                            │
                                            ▼
                           MediaPipe 21 Hand Landmarks Tracking
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
     Feature Engineering Pipeline                                Dual-Buffer Engine
   82 Invariant Spatial Features                               (Canvas Color + Mask)
   (63 Norm Coords + 14 Dist Ratios + 5 Angles)                          │
               │                                                         │
   ┌───────────┴───────────┐                                             │
   ▼                       ▼                                             │
[ML Classifier]     [DeepNet DNN]                                        │
Random Forest       4-Layer Dense                                        │
(100% Accuracy)    (100% Accuracy)                                       │
   │                       │                                             │
   └───────────┬───────────┘                                             │
               ▼                                                         ▼
       Active Mode Decision ─────────────────────────────────► Stroke Rendering & EMA
  (DRAW / SELECT / ERASE / STANDBY)                             Coordinate Smoothing
               │                                                         │
               ▼                                                         ▼
   StrokeCNN Sketch Recognizer ◄────────────────────────────── Telemetry & Analytics
   (Circle / Rectangle / Triangle)                             (Speed, Jitter, FPS)
```

---

## 2. The Four Proportionate AI Pillars

### Pillar 1: Datasets (`dataset/`)
- **Benchmark Hand Gesture Dataset (`dataset/hand_gestures.csv`)**:
  - **2,500 balanced samples** (500 samples per class across `DRAW`, `SELECT`, `ERASE`, `STANDBY`, `CLEAR`).
  - **82 engineered columns**:
    - 63 wrist-relative, scale-normalized 3D coordinates ($21 \times 3$).
    - 14 pairwise Euclidean distance ratios (curl and spread metrics).
    - 5 joint bend cosine angles (PIP flexions).
- **Air-Drawn Shapes Dataset (`dataset/air_shapes_dataset.npz`)**:
  - **1,500 normalized single-channel 28x28 sketches** across 5 shapes: `CIRCLE`, `RECTANGLE`, `TRIANGLE`, `STAR`, `LINE`.
- **Live Dataset Collector (`dataset/dataset_collector.py`)**:
  - Interactive webcam tool for capturing custom gesture samples with live HUD feedback.

---

### Pillar 2: Machine Learning (`ml_models/`)
- **Feature Invariance**:
  - Normalizes landmark coordinates with respect to the wrist joint $(x_0, y_0, z_0)$ and divides by palm scale $\lVert \text{MCP}_9 - \text{Wrist} \rVert$.
  - Invariant to hand distance, screen position, and user hand sizes.
- **Model Benchmark Results (5-Fold Stratified Cross-Validation)**:

| Model Architecture | 5-Fold CV Accuracy | Test Accuracy | Test F1-Score | Inference Latency |
|---|---|---|---|---|
| **Random Forest (Champion)** | **100.00%** | **100.00%** | **100.00%** | **0.03 ms** |
| Support Vector Machine (RBF) | 100.00% | 100.00% | 100.00% | 0.02 ms |
| K-Nearest Neighbors ($k=5$) | 100.00% | 100.00% | 100.00% | 0.01 ms |
| Logistic Regression | 100.00% | 100.00% | 100.00% | 0.005 ms |

- **Champion Model Deployment**:
  - Saved to `ml_models/best_gesture_model.pkl`.
  - Runs in real time in `main.py` with sub-millisecond per-frame inference.

---

### Pillar 3: Deep Learning (`dl_models/`)
1. **Deep Landmark Neural Network (`DeepGestureNet`)**:
   - **Architecture**: Input (82) $\to$ Dense (128) + ReLU + Dropout $\to$ Dense (64) + ReLU + Dropout $\to$ Dense (32) + ReLU $\to$ Dense (5) + Softmax.
   - **Performance**: 100.00% validation accuracy, 0.008 ms latency.
   - **Weights**: Saved to `dl_models/deep_gesture_weights.npz`.
2. **Air-Canvas Sketch Recognizer (`StrokeCNN`)**:
   - Classifies user air drawings (circles, rectangles, triangles, stars, lines) from canvas stroke masks.
   - **Performance**: 98.67% test accuracy.
   - **Weights**: Saved to `dl_models/stroke_cnn_model.pkl`.
   - **Agentic AI Integration**: Press `[R]` or click `RECOGNIZE` on the toolbar to identify what you drew in the air!

---

### Pillar 4: Data Analytics (`analytics/`)
- **Exploratory Data Analysis (EDA)**:
  - `analytics/plots/eda_class_distribution.png`: Sample count per gesture.
  - `analytics/plots/eda_pca_clusters.png`: 2D PCA latent cluster projections.
  - `analytics/plots/eda_feature_correlation.png`: Correlation heatmap across 82 geometric features.
  - `analytics/plots/eda_finger_distances.png`: Fingertip-to-wrist distance boxplots.
- **Model Evaluation Analytics**:
  - `analytics/plots/model_comparison_metrics.png`: Grouped comparative bar chart of Accuracy and F1 scores.
  - `analytics/plots/latency_vs_accuracy.png`: Latency vs. accuracy trade-off frontier.
  - `analytics/plots/confusion_matrices.png`: Test set confusion matrix.
- **Real-Time Session Telemetry Analytics**:
  - `analytics/plots/session_summary_dashboard.png`: Quantifies drawing speed, mode breakdown pie chart, FPS stability, and **76.5% tremor reduction** via the EMA filter.
  - Telemetry CSV exported to `saved_drawings/session_telemetry.csv`.
- **Full Quantitative Report**:
  - Detailed findings documented in [`analytics/ANALYTICS_REPORT.md`](file:///c:/Users/geeth/OneDrive/Desktop/Agentic%20Ai%20enabled%20real%20time%20air%20drawing%20application/analytics/ANALYTICS_REPORT.md).

---

## 3. Gesture Controls & User Interface

| Gesture | Mode | Function |
|---|---|---|
| ☝️ **Index finger UP** | `DRAW` | Draws smooth continuous strokes on the canvas |
| ✌️ **Index + Middle fingers UP** | `SELECT` | Moves cursor without drawing; hovers over buttons to select |
| ✊ **Closed Fist** | `ERASE` | Activates circular eraser to remove strokes |
| 🖐️ **Splayed Open Palm** | `CLEAR` | Instantly clears the canvas |
| ✋ **Relaxed Hand** | `STANDBY` | Standby mode (no drawing action) |

### On-Screen Toolbar Controls
- **Colors**: `BLUE`, `GREEN`, `RED`, `BLACK`
- **Tools**: `ERASER`, `CLEAR`, `SIZE`, `SAVE`
- **AI Controls**:
  - `AI: ML/DL`: Toggles between Machine Learning (Random Forest) and Deep Learning (DeepNet).
  - `RECOGNIZE`: Runs StrokeCNN to recognize shapes drawn on the canvas.
  - `ANALYTICS`: Saves session telemetry and exports the visual analytics dashboard.

### Keyboard Shortcuts
- `1` / `2` / `3` / `4`: Select Blue, Green, Red, or Black
- `M`: Toggle AI Engine (ML / DL / Heuristic)
- `R`: AI Recognize current drawing
- `A`: Export Data Analytics Dashboard
- `C`: Clear Canvas
- `S`: Save Drawing (PNG)
- `+` / `-`: Increase / Decrease Brush Size
- `Q` or `ESC`: Quit application and automatically save telemetry

---

## 4. Project Directory Structure

```
Agentic Ai enabled real time air drawing application/
├── dataset/
│   ├── __init__.py
│   ├── feature_extractor.py       # 82-dimensional invariant feature engineering
│   ├── dataset_generator.py       # Generates benchmark gesture dataset (2,500 samples)
│   ├── dataset_collector.py       # Live webcam gesture recorder
│   ├── shapes_dataset.py          # Air-drawn sketch dataset (1,500 samples)
│   ├── hand_gestures.csv          # 2,500 samples x 83 columns
│   └── air_shapes_dataset.npz     # 1,500 sketch rasters + labels
├── ml_models/
│   ├── __init__.py
│   ├── train_ml.py                # Trains RFC, SVM, KNN, Logistic Regression
│   ├── ml_gesture_classifier.py   # Real-time sub-millisecond inference engine
│   ├── best_gesture_model.pkl     # Champion Random Forest model
│   ├── label_encoder.pkl          # Target label encoder
│   └── model_metadata.json        # 5-fold CV metrics and benchmark report
├── dl_models/
│   ├── __init__.py
│   ├── deep_gesture_net.py        # 4-layer Deep Landmark Neural Network (DNN)
│   ├── stroke_cnn.py              # Air-canvas sketch & shape recognizer
│   ├── train_dl.py                # Deep learning training pipeline
│   ├── deep_gesture_weights.npz   # Trained DeepNet weights
│   ├── stroke_cnn_model.pkl       # Trained sketch classifier model
│   └── dl_metadata.json           # DL metrics and convergence history
├── analytics/
│   ├── __init__.py
│   ├── dataset_analytics.py       # EDA, class balance, PCA, correlation heatmaps
│   ├── model_evaluation_analytics.py # Performance comparisons and confusion matrix
│   ├── session_analytics.py       # Telemetry tracking & tremor reduction analytics
│   ├── run_all_analytics.py       # Master runner generating all reports & figures
│   ├── ANALYTICS_REPORT.md        # Comprehensive quantitative analytics report
│   └── plots/                     # High-resolution generated PNG figures
├── saved_drawings/                # Exported drawings and session telemetry CSVs
├── hand_tracking.py               # MediaPipe hand landmark detection wrapper
├── gestures.py                    # Heuristic pose classifier & EMA smooth pointer
├── drawing.py                     # Dual-buffer canvas manager
├── ai_interface.py                # Agentic AI natural language dispatcher
├── main.py                        # Real-time application event loop
├── test_app.py                    # 10-point automated integration test suite
├── requirements.txt               # Project dependencies
└── run.bat                        # Interactive Windows launcher menu
```

---

## 5. Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: One-Click Execution via `run.bat`
Double-click `run.bat` or run:
```bat
run.bat
```
The interactive menu allows you to:
1. Launch Real-Time Air Drawing Application
2. Train Machine Learning Models
3. Train Deep Learning Models
4. Generate Data Analytics Dashboard & Reports
5. Collect Custom Gesture Dataset
6. Run 10-Point Integration Tests

### Or run individually via terminal:
```bash
# 1. Run the application
python main.py

# 2. Train ML models
python ml_models/train_ml.py

# 3. Train DL models
python dl_models/train_dl.py

# 4. Generate all analytics figures
python analytics/run_all_analytics.py

# 5. Run test suite
python test_app.py
```
