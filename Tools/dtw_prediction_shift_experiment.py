import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from Tools.DatasetConverter import DatasetConverter


def load_labelled_dataframe(csv_path: str) -> pd.DataFrame:
    """Load CSV and generate turning-point labels if not present."""
    dc = DatasetConverter(file_path=csv_path, save_path=None)
    df = dc.convert(label_type=1, volume=True)
    return df


def build_windows(df: pd.DataFrame, window_size: int = 600):
    """Construct sliding windows and center labels."""
    feature_cols = ["Close", "High", "Low", "Open", "Volume"]
    half_w = window_size // 2
    X_list, y_list = [], []
    for start_idx in range(len(df) - window_size + 1):
        end_idx = start_idx + window_size
        window = df.iloc[start_idx:end_idx][feature_cols].values
        label = df.iloc[start_idx + half_w]["Labels"]
        X_list.append(window)
        y_list.append(label)
    return np.array(X_list), np.array(y_list)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    dist = 0.0
    for k in range(a.shape[1]):
        d, _ = fastdtw(a[:, k], b[:, k])
        dist += d
    return dist

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two windows."""
    return np.linalg.norm(a - b, axis=0).sum()

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two windows."""
    dot_product = np.dot(a.flatten(), b.flatten())
    norm_a = np.linalg.norm(a.flatten())
    norm_b = np.linalg.norm(b.flatten())
    if norm_a == 0 or norm_b == 0:
        return float('inf')  # Avoid division by zero
    return 1 - (dot_product / (norm_a * norm_b))

def compute_shift_distances(X: np.ndarray, y: np.ndarray, shift: int = 3):
    indices = np.where(y == 1)[0]
    results = []
    for idx in indices:
        base = X[idx]
        for off in range(-shift, shift + 1):
            if idx + off < 0 or idx + off >= len(X):
                continue
            comp = X[idx + off]
            dtw_dist = dtw_distance(base, comp)
            euc_dist = euclidean_distance(base, comp)
            cos_dist = cosine_distance(base, comp)
            results.append({"offset": off, "dtw": dtw_dist, "euclidean": euc_dist, "cosine": cos_dist})
    return pd.DataFrame(results)

def compute_tp_pair_distances(
    X: np.ndarray, y: np.ndarray, num_pairs: int = 5, min_separation: int = 5
) -> list:
    """Compute DTW distances between several true-positive window pairs."""
    indices = np.where(y == 1)[0]
    if len(indices) < 2:
        return []

    # Build eligible pairs with a minimum separation to avoid overlap
    eligible = [
        (i, j)
        for i in indices
        for j in indices
        if j > i and (j - i) > min_separation
    ]
    if not eligible:
        return []

    rng = np.random.default_rng(0)
    chosen = rng.choice(len(eligible), size=min(num_pairs, len(eligible)), replace=False)
    dists = []
    for idx in chosen:
        i, j = eligible[idx]
        dists.append([dtw_distance(X[i], X[j]), euclidean_distance(X[i], X[j]), cosine_distance(X[i], X[j])])
    return dists


def main():
    csv_path = os.path.join("..", "Dataset", "ftse_minute_data_daily.csv")
    df = load_labelled_dataframe(csv_path)
    X, y = build_windows(df, window_size=600)
    df_dist = compute_shift_distances(X, y, shift=5)
    pivot = df_dist.pivot_table(index="offset", values="dtw", aggfunc="mean")
    pivot.plot(kind="bar", rot=0)
    plt.xlabel("Offset from true turning point")
    plt.ylabel("Average DTW distance")
    plt.title("DTW distance vs. offset")
    plt.tight_layout()
    plt.savefig("dtw_shift_distances.png")

    plt.xlabel("Offset from true turning point")
    plt.ylabel("Average Euclidean distance")
    pivot = df_dist.pivot_table(index="offset", values="euclidean", aggfunc="mean")
    pivot.plot(kind="bar", rot=0)
    plt.title("Euclidean distance vs. offset")
    plt.tight_layout()
    plt.savefig("euclidean_shift_distances.png")

    plt.xlabel("Offset from true turning point")
    plt.ylabel("Average Cosine distance")
    pivot = df_dist.pivot_table(index="offset", values="cosine", aggfunc="mean")
    pivot.plot(kind="bar", rot=0)
    plt.title("Cosine distance vs. offset")
    plt.tight_layout()
    plt.savefig("cosine_shift_distances.png")
    print(pivot)

    tp_dists = compute_tp_pair_distances(X, y, num_pairs=5, min_separation=5)
    if tp_dists:
        print("True-positive pair distances:", tp_dists)
        print("Average TP distance:", np.mean(tp_dists))

if __name__ == "__main__":
    main()

