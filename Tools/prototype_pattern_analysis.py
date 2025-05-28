import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Tools.DatasetConverter import DatasetConverter
from PrototypeBasedModel import PrototypeSelector, PrototypeFeatureExtractor

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'ftse_minute_data_daily.csv')
WINDOW_SIZE = 600
N_PROTOTYPES = 5
SHORT_MA = 30
LONG_MA = 120


def build_windows(df: pd.DataFrame, window_size: int = 600):
    feature_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    half = window_size // 2
    X_list, y_list = [], []
    for start in range(len(df) - window_size + 1):
        end = start + window_size
        window = df.iloc[start:end][feature_cols].values
        label = df.iloc[start + half]['Labels']
        X_list.append(window)
        y_list.append(label)
    return np.array(X_list), np.array(y_list)


def detect_ma_crossovers(series: np.ndarray, short_window: int, long_window: int):
    short_ma = pd.Series(series).rolling(short_window).mean()
    long_ma = pd.Series(series).rolling(long_window).mean()
    crossover_up = np.where((short_ma.shift(1) < long_ma.shift(1)) & (short_ma >= long_ma))[0]
    crossover_down = np.where((short_ma.shift(1) > long_ma.shift(1)) & (short_ma <= long_ma))[0]
    return crossover_up, crossover_down


def plot_prototype_patterns(protos: np.ndarray, save_dir: str = 'figures'):
    os.makedirs(save_dir, exist_ok=True)
    for idx, proto in enumerate(protos):
        close = proto[:, 0]
        short_ma = pd.Series(close).rolling(SHORT_MA).mean()
        long_ma = pd.Series(close).rolling(LONG_MA).mean()
        max_idx = argrelextrema(close, np.greater, order=5)[0]
        min_idx = argrelextrema(close, np.less, order=5)[0]
        cross_up, cross_down = detect_ma_crossovers(close, SHORT_MA, LONG_MA)

        plt.figure(figsize=(8, 4))
        plt.plot(close, label='Close', color='black')
        plt.plot(short_ma, '--', label=f'MA{SHORT_MA}')
        plt.plot(long_ma, '-.', label=f'MA{LONG_MA}')
        if len(max_idx) > 0:
            plt.scatter(max_idx, close[max_idx], color='red', marker='^', label='Local Max')
        if len(min_idx) > 0:
            plt.scatter(min_idx, close[min_idx], color='green', marker='v', label='Local Min')
        if len(cross_up) > 0:
            plt.scatter(cross_up, close[cross_up], color='blue', marker='o', label='MA Cross Up')
        if len(cross_down) > 0:
            plt.scatter(cross_down, close[cross_down], color='purple', marker='x', label='MA Cross Down')
        plt.axvline(len(close) // 2, color='red', ls=':')
        plt.title(f'Prototype {idx}')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prototype_pattern_{idx}.png'))
        plt.close()


def summarize_prototypes(protos: np.ndarray):
    summaries = []
    for idx, proto in enumerate(protos):
        close = proto[:, 0]
        max_idx = argrelextrema(close, np.greater, order=5)[0]
        min_idx = argrelextrema(close, np.less, order=5)[0]
        cross_up, cross_down = detect_ma_crossovers(close, SHORT_MA, LONG_MA)
        slope = close[-1] - close[0]
        trend = 'uptrend' if slope > 0 else 'downtrend' if slope < 0 else 'sideways'
        summaries.append({
            'prototype': idx,
            'num_maxima': len(max_idx),
            'num_minima': len(min_idx),
            'num_cross_up': len(cross_up),
            'num_cross_down': len(cross_down),
            'trend': trend,
        })
    df = pd.DataFrame(summaries)
    print('Prototype Pattern Summary:')
    print(df.to_string(index=False))


def main():
    dc = DatasetConverter(file_path=DATA_PATH, save_path=None)
    df = dc.convert(label_type=1, volume=True)
    X, y = build_windows(df, window_size=WINDOW_SIZE)

    selector = PrototypeSelector(X, y, window_size=WINDOW_SIZE)
    protos, _, _, _ = selector.select_prototypes(
        num_prototypes=N_PROTOTYPES,
        selection_type='positive_only',
    )

    plot_prototype_patterns(protos, save_dir='figures')
    summarize_prototypes(protos)

    # Map windows to prototypes and show best matches
    extractor = PrototypeFeatureExtractor(torch.from_numpy(X), torch.from_numpy(protos))
    distances = extractor._compute_euclidean_features().mean(dim=2).numpy()
    for idx in range(N_PROTOTYPES):
        closest = np.argsort(distances[:, idx])[:3]
        for j, w_idx in enumerate(closest):
            window = X[w_idx]
            close = window[:, 0]
            cross_up, cross_down = detect_ma_crossovers(close, SHORT_MA, LONG_MA)
            plt.figure(figsize=(8, 4))
            plt.plot(close, label='Window')
            plt.plot(protos[idx, :, 0], label=f'Prototype {idx}', linestyle='--')
            if len(cross_up) > 0:
                plt.scatter(cross_up, close[cross_up], color='blue', marker='o', label='MA Cross Up')
            if len(cross_down) > 0:
                plt.scatter(cross_down, close[cross_down], color='purple', marker='x', label='MA Cross Down')
            plt.axvline(len(close) // 2, color='red', ls=':')
            plt.title(f'Prototype {idx} - Match {j}')
            plt.xlabel('Time Step')
            plt.ylabel('Normalized Price')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join('figures', f'proto_{idx}_match_{j}.png'))
            plt.close()


if __name__ == '__main__':
    main()
