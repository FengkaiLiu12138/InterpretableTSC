import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Tools.DatasetConverter import DatasetConverter

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'ftse_minute_data_daily.csv')
WINDOW_SIZE = 600
FEATURE_COLS = ["Close", "High", "Low", "Open", "Volume"]


def _load_dataframe() -> pd.DataFrame:
    dc = DatasetConverter(file_path=DATA_PATH, save_path=None)
    return dc.convert(label_type=1, volume=True)


def _build_windows(df: pd.DataFrame, window_size: int = WINDOW_SIZE):
    half = window_size // 2
    X_list, y_list = [], []
    for start in range(len(df) - window_size + 1):
        end = start + window_size
        window = df.iloc[start:end][FEATURE_COLS].values
        label = df.iloc[start + half]["Labels"]
        X_list.append(window)
        y_list.append(label)
    return np.array(X_list), np.array(y_list)


def _sample_windows(X: np.ndarray, y: np.ndarray, num_each: int = 2, seed: int | None = None):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < num_each or len(neg_idx) < num_each:
        raise ValueError("Not enough positive or negative samples to draw from")
    chosen_pos = rng.choice(pos_idx, num_each, replace=False)
    chosen_neg = rng.choice(neg_idx, num_each, replace=False)
    indices = np.concatenate([chosen_pos, chosen_neg])
    labels = np.array([1] * num_each + [0] * num_each)
    windows = X[indices]
    return windows, labels


def _plot_windows(windows: np.ndarray, labels: np.ndarray, out_dir: str = 'figures') -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, (window, label) in enumerate(zip(windows, labels), 1):
        plt.figure(figsize=(8, 2))
        plt.plot(window[:, 0], label='Close')
        plt.axvline(WINDOW_SIZE // 2, color='r', ls='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'window_{i}_label_{label}.png'))
        plt.close()


def main() -> None:
    df = _load_dataframe()
    X, y = _build_windows(df, window_size=WINDOW_SIZE)
    windows, labels = _sample_windows(X, y, num_each=2, seed=42)
    _plot_windows(windows, labels)


if __name__ == '__main__':
    main()
