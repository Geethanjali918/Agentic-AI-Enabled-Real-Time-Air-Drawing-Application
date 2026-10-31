"""
analytics/dataset_analytics.py
------------------------------
Exploratory Data Analysis (EDA) & Statistical Profiling of the Air Drawing Hand Gesture Dataset.
Generates publication-quality analytics visualizations:
  1. Class Distribution Bar Chart (eda_class_distribution.png)
  2. 2D Principal Component Analysis (PCA) Cluster Projection (eda_pca_clusters.png)
  3. Feature Correlation Heatmap (eda_feature_correlation.png)
  4. Fingertip-to-Wrist Distance Boxplots (eda_finger_distances.png)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def run_dataset_eda(dataset_path="dataset/hand_gestures.csv", output_dir="analytics/plots"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Dataset Analytics] Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # 1. Class Distribution
    print("[Dataset Analytics] Generating Class Distribution Chart...")
    counts = df["gesture"].value_counts()
    colors = ["#2b5c8f", "#2e7d32", "#c62828", "#f57f17", "#6a1b9a"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=colors[:len(counts)], width=0.55, edgecolor="black", linewidth=1.2)
    plt.title("Hand Gesture Dataset - Class Sample Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Gesture Class", fontsize=12, labelpad=10)
    plt.ylabel("Sample Count", fontsize=12, labelpad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 10, f"{int(h)}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "eda_class_distribution.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"  -> Saved {chart1_path}")

    # 2. PCA 2D Clustering
    print("[Dataset Analytics] Generating 2D PCA Cluster Projection...")
    feature_cols = [c for c in df.columns if c not in ["gesture", "timestamp", "sample_id"]]
    X = df[feature_cols].values
    y = df["gesture"].values

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(9, 6))
    gesture_labels = sorted(df["gesture"].unique())
    palette = {"DRAW": "#1976d2", "SELECT": "#388e3c", "ERASE": "#d32f2f", "STANDBY": "#fbc02d", "CLEAR": "#7b1fa2"}

    for g in gesture_labels:
        mask = (y == g)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=g, color=palette.get(g, "#888888"), alpha=0.65, s=35, edgecolors="none")

    plt.title(f"2D PCA Projection of 82-Dim Landmark Features\n(PC1: {explained_var[0]:.1f}%, PC2: {explained_var[1]:.1f}% Variance Explained)", fontsize=13, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({explained_var[0]:.1f}%)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({explained_var[1]:.1f}%)", fontsize=11)
    plt.legend(title="Gesture", loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "eda_pca_clusters.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"  -> Saved {chart2_path}")

    # 3. Fingertip-to-Wrist Distance Boxplot
    print("[Dataset Analytics] Generating Fingertip Distance Boxplots...")
    dist_cols = [c for c in df.columns if "dist_wrist_tip" in c]
    if dist_cols:
        plt.figure(figsize=(10, 5.5))
        # Plot Index Fingertip distance (tip 8) across gestures
        col_index = "dist_wrist_tip_8" if "dist_wrist_tip_8" in df.columns else dist_cols[1]
        data_to_plot = [df[df["gesture"] == g][col_index].values for g in gesture_labels]

        box = plt.boxplot(data_to_plot, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
        plt.xticks(range(1, len(gesture_labels) + 1), gesture_labels)
        for patch, g in zip(box["boxes"], gesture_labels):
            patch.set_facecolor(palette.get(g, "#90caf9"))
            patch.set_alpha(0.8)

        plt.title("Normalized Index Fingertip-to-Wrist Distance Across Gestures", fontsize=13, fontweight="bold")
        plt.xlabel("Gesture Class", fontsize=11)
        plt.ylabel("Normalized Distance (Palm Heights)", fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        chart3_path = os.path.join(output_dir, "eda_finger_distances.png")
        plt.savefig(chart3_path, dpi=300)
        plt.close()
        print(f"  -> Saved {chart3_path}")

    # 4. Feature Correlation Matrix
    print("[Dataset Analytics] Generating Correlation Matrix Heatmap...")
    sample_feats = [c for c in feature_cols if "dist_" in c or "angle_" in c][:12]
    corr = df[sample_feats].corr().values

    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Geometric & Angle Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(range(len(sample_feats)), sample_feats, rotation=45, ha="right", fontsize=9)
    plt.yticks(range(len(sample_feats)), sample_feats, fontsize=9)
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "eda_feature_correlation.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"  -> Saved {chart4_path}")

    summary_stats = df[feature_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    summary_path = os.path.join(output_dir, "feature_statistical_summary.csv")
    summary_stats.to_csv(summary_path)
    print(f"[Dataset Analytics] Summary statistics saved to {summary_path}")

    return {
        "total_samples": len(df),
        "num_features": len(feature_cols),
        "class_counts": counts.to_dict(),
        "pca_variance_explained": [round(explained_var[0], 2), round(explained_var[1], 2)]
    }


if __name__ == "__main__":
    run_dataset_eda()
