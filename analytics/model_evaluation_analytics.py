"""
analytics/model_evaluation_analytics.py
---------------------------------------
Comparative Model Evaluation & Performance Analytics for ML & DL models.
Generates:
  1. Multi-Model Performance Comparison (model_comparison_metrics.png)
  2. Latency vs. Accuracy Trade-Off Frontier (latency_vs_accuracy.png)
  3. Side-by-Side Confusion Matrices (confusion_matrices.png)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_model_analytics(ml_meta_path="ml_models/model_metadata.json",
                        dl_meta_path="dl_models/dl_metadata.json",
                        output_dir="analytics/plots"):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ml_meta_path):
        print(f"[Model Analytics] {ml_meta_path} not found.")
        return

    with open(ml_meta_path, "r") as f:
        ml_data = json.load(f)

    # 1. Gather model metrics
    models = []
    accs = []
    f1s = []
    latencies = []

    for name, m in ml_data["results"].items():
        models.append(name)
        accs.append(m["test_accuracy"] * 100)
        f1s.append(m["test_f1"] * 100)
        latencies.append(m["latency_ms"])

    # Add Deep Learning model if available
    if os.path.exists(dl_meta_path):
        with open(dl_meta_path, "r") as f:
            dl_data = json.load(f)
        models.append("DeepNet (DNN)")
        accs.append(dl_data["val_accuracy"] * 100)
        f1s.append(dl_data["val_accuracy"] * 100)
        latencies.append(dl_data["inference_latency_ms"])

    # 1. Bar Chart Comparison
    print("[Model Analytics] Generating Model Metrics Comparison Chart...")
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 5.5))
    rects1 = plt.bar(x - width/2, accs, width, label="Accuracy (%)", color="#1976d2", edgecolor="black")
    rects2 = plt.bar(x + width/2, f1s, width, label="F1-Score (%)", color="#388e3c", edgecolor="black")

    plt.title("Comparative Performance Across ML & DL Models", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Model Architecture", fontsize=12, labelpad=10)
    plt.ylabel("Score (%)", fontsize=12, labelpad=10)
    plt.xticks(x, models, fontsize=10)
    plt.ylim(90, 105)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for r in rects1:
        plt.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.5, f"{r.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects2:
        plt.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.5, f"{r.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "model_comparison_metrics.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"  -> Saved {chart1_path}")

    # 2. Latency vs Accuracy Frontier
    print("[Model Analytics] Generating Latency vs. Accuracy Frontier Plot...")
    plt.figure(figsize=(9, 5.5))
    scatter_colors = ["#e53935", "#1e88e5", "#43a047", "#8e24aa", "#fb8c00"]

    for i, m_name in enumerate(models):
        plt.scatter(latencies[i], accs[i], s=180, color=scatter_colors[i % len(scatter_colors)], edgecolors="black", linewidth=1.5, zorder=5)
        plt.annotate(
            f"  {m_name}\n  ({latencies[i]:.2f}ms, {accs[i]:.1f}%)",
            (latencies[i], accs[i]),
            textcoords="offset points",
            xytext=(10, -5),
            fontsize=9,
            fontweight="bold"
        )

    plt.title("Model Trade-off: Inference Latency vs. Accuracy", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Inference Latency per Frame (milliseconds)", fontsize=11)
    plt.ylabel("Classification Accuracy (%)", fontsize=11)
    plt.ylim(95, 102)
    plt.xlim(-0.01, max(latencies) * 1.5 + 0.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "latency_vs_accuracy.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"  -> Saved {chart2_path}")

    # 3. Confusion Matrix of Champion Model
    best_name = ml_data.get("best_model_name", "Random Forest")
    if best_name in ml_data["results"]:
        print(f"[Model Analytics] Generating Confusion Matrix for Champion Model '{best_name}'...")
        cm = np.array(ml_data["results"][best_name]["confusion_matrix"])
        classes = ml_data["class_names"]

        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Confusion Matrix: {best_name} (Test Set)", fontsize=13, fontweight="bold", pad=12)
        plt.colorbar(fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha="right")
        plt.yticks(tick_marks, classes)

        # Labels
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black", fontweight="bold")

        plt.ylabel("True Class", fontsize=11)
        plt.xlabel("Predicted Class", fontsize=11)
        plt.tight_layout()
        chart3_path = os.path.join(output_dir, "confusion_matrices.png")
        plt.savefig(chart3_path, dpi=300)
        plt.close()
        print(f"  -> Saved {chart3_path}")


if __name__ == "__main__":
    run_model_analytics()
