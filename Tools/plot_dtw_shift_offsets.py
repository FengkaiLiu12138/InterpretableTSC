import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Tools.DatasetConverter import DatasetConverter


def load_labelled_dataframe(csv_path: str) -> pd.DataFrame:
    """Load CSV and generate turning-point labels if not present."""
    dc = DatasetConverter(file_path=csv_path, save_path=None)
    return dc.convert(label_type=1, volume=True)


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
    """Compute DTW distance across all features."""
    dist = 0.0
    for k in range(a.shape[1]):
        d, _ = fastdtw(a[:, k], b[:, k], radius=1)
        dist += d
    return dist


def compute_shift_distances(X: np.ndarray, y: np.ndarray, shift: int = 5, max_tp: int = 10) -> pd.DataFrame:
    """Compute DTW distance between each TP window and its shifted neighbours."""
    indices = np.where(y == 1)[0][:max_tp]
    results = []
    for idx in indices:
        base = X[idx]
        for off in range(-shift, shift + 1):
            if idx + off < 0 or idx + off >= len(X):
                continue
            comp = X[idx + off]
            results.append({"offset": off, "dtw": dtw_distance(base, comp)})
    return pd.DataFrame(results)


def compute_tp_pair_distances(
    X: np.ndarray, y: np.ndarray, num_pairs: int = 5, min_separation: int = 5
) -> list:
    """Compute DTW distances between several true-positive window pairs."""
    indices = np.where(y == 1)[0]
    if len(indices) < 2:
        return []
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
        dists.append(dtw_distance(X[i], X[j]))
    return dists


def main() -> None:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "Dataset", "ftse_minute_data_daily.csv")
    df = load_labelled_dataframe(csv_path)
    X, y = build_windows(df, window_size=600)

    shift_df = compute_shift_distances(X, y, shift=5, max_tp=10)
    pivot = shift_df.pivot_table(index="offset", values="dtw", aggfunc="mean")

    tp_dists = compute_tp_pair_distances(X, y, num_pairs=5, min_separation=5)
    mean_tp_dtw = np.mean(tp_dists) if tp_dists else 0.0

    plt.figure(figsize=(6, 4))
    bars = plt.bar(pivot.index, pivot["dtw"], color="skyblue")
    for b in bars:
        height = b.get_height()
        plt.text(
            b.get_x() + b.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.axhline(mean_tp_dtw, color="r", ls="--", label=f"Avg TP DTW = {mean_tp_dtw:.2f}")
    plt.xlabel("Offset from true turning point")
    plt.ylabel("Average DTW distance")
    plt.title("DTW distance vs offset")
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "dtw_shift_offsets.png"))
    plt.close()
    print(pivot)
    print("Average DTW distance between TP pairs:", mean_tp_dtw)


if __name__ == "__main__":
    main()
