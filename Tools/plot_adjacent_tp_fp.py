import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Tools.DatasetConverter import DatasetConverter

DATA_PATH = os.path.join('..', 'Dataset', 'ftse_minute_data_daily.csv')
WINDOW_SIZE = 600
TP_INDEX = 805
FP_INDEX = 807

FEATURE_COLS = ["Close", "High", "Low", "Open", "Volume"]


def _load_dataframe():
    dc = DatasetConverter(file_path=DATA_PATH, save_path=None)
    return dc.convert(label_type=1, volume=True)


def _extract_window(df: pd.DataFrame, idx: int) -> np.ndarray:
    half = WINDOW_SIZE // 2
    start = idx - half
    end = idx + half + 1
    if start < 0 or end > len(df):
        raise IndexError("Index out of bounds for window extraction")
    return df.iloc[start:end][FEATURE_COLS].values


def plot_adjacent_windows(out_path: str = os.path.join('figures', 'tp_fp_adjacent.png')) -> None:
    df = _load_dataframe()
    tp_window = _extract_window(df, TP_INDEX)
    fp_window = _extract_window(df, FP_INDEX)
    half = WINDOW_SIZE // 2

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    axes[0].plot(tp_window[:, 0], label='Close')
    axes[0].axvline(half, color='r', ls='--', label='Turning point')
    axes[0].set_title(f'True Positive Window (index {TP_INDEX})')
    axes[0].legend()

    axes[1].plot(fp_window[:, 0], label='Close')
    axes[1].axvline(half, color='r', ls='--', label='Window center')
    axes[1].set_title(f'False Positive Window (index {FP_INDEX})')
    axes[1].legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == '__main__':
    plot_adjacent_windows()