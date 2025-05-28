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


def plot_prototype_extrema(protos: np.ndarray, save_dir: str, prefix: str = 'proto_extrema'):
    os.makedirs(save_dir, exist_ok=True)
    for idx, proto in enumerate(protos):
        close = proto[:, 0]
        short_ma = pd.Series(close).rolling(30).mean()
        long_ma = pd.Series(close).rolling(120).mean()
        max_idx = argrelextrema(close, np.greater, order=5)[0]
        min_idx = argrelextrema(close, np.less, order=5)[0]

        plt.figure(figsize=(8, 4))
        plt.plot(close, label='Close', color='black')
        plt.plot(short_ma, '--', label='MA30')
        plt.plot(long_ma, '-.', label='MA120')
        plt.scatter(max_idx, close[max_idx], color='red', marker='^', label='Local Max')
        plt.scatter(min_idx, close[min_idx], color='green', marker='v', label='Local Min')
        plt.axvline(len(close) // 2, color='red', ls=':')
        plt.title(f'Prototype {idx}')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_{idx}.png'))
        plt.close()


def main():
    dc = DatasetConverter(file_path=DATA_PATH, save_path=None)
    df = dc.convert(label_type=1, volume=True)
    X, y = build_windows(df, window_size=WINDOW_SIZE)

    selector = PrototypeSelector(X, y, window_size=WINDOW_SIZE)
    protos, _, _, _ = selector.select_prototypes(
        num_prototypes=N_PROTOTYPES,
        selection_type='positive_only'
    )

    extractor = PrototypeFeatureExtractor(torch.from_numpy(X), torch.from_numpy(protos))
    extractor.plot_prototype_cycles(
        short_window=30,
        long_window=120,
        save_dir='figures',
        prefix='prototype_cycle'
    )

    plot_prototype_extrema(protos, 'figures')


if __name__ == '__main__':
    main()
