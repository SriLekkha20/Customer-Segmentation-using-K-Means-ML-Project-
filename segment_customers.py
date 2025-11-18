"""
Customer segmentation using K-Means clustering.

This script:
- Reads customer data from data/customers.csv
- Scales selected numeric features
- Uses K-Means to cluster customers
- Saves the results and generates simple plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/customers.csv")
OUTPUT_PATH = Path("data/customers_with_clusters.csv")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found at {path}. Please place your customers.csv there."
        )
    return pd.read_csv(path)


def compute_elbow_curve(features_scaled: np.ndarray, k_min: int = 2, k_max: int = 10):
    inertias = []
    k_values = list(range(k_min, k_max + 1))

    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(features_scaled)
        inertias.append(model.inertia_)

    plt.figure()
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow Curve for K-Means")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_clusters(df: pd.DataFrame, cluster_col: str):
    plt.figure()
    for cluster_id in sorted(df[cluster_col].unique()):
        subset = df[df[cluster_col] == cluster_id]
        plt.scatter(
            subset["AnnualIncome"],
            subset["SpendingScore"],
            alpha=0.7,
            label=f"Cluster {cluster_id}",
        )

    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.title("Customer Segments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} customer records.")

    # Choose numeric features for clustering
    feature_cols = ["Age", "AnnualIncome", "SpendingScore"]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in dataset: {missing}")

    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional: show elbow curve to manually choose k
    compute_elbow_curve(X_scaled, k_min=2, k_max=8)

    # For now, fix k=4 (can be tuned)
    k = 4
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df["Cluster"] = cluster_labels
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved clustered data to {OUTPUT_PATH}")

    visualize_clusters(df, "Cluster")


if __name__ == "__main__":
    main()
