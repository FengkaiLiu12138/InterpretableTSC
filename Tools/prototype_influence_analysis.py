import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from Tools.DatasetConverter import DatasetConverter
from PrototypeBasedModel import PrototypeSelector, PrototypeFeatureExtractor

# Parameters
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'ftse_minute_data_daily.csv')
WINDOW_SIZE = 600
N_PROTOTYPES = 5
TEST_RATIO = 0.2


def build_windows(df: pd.DataFrame, window_size: int = WINDOW_SIZE):
    feature_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    half = window_size // 2
    X_list, y_list = [], []
    for start in range(len(df) - window_size + 1):
        end = start + window_size
        window = df.iloc[start:end][feature_cols].values
        label = df.iloc[start + half]['Labels']
        X_list.append(window)
        y_list.append(label)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def main():
    # Load and convert dataset
    dc = DatasetConverter(file_path=DATA_PATH, save_path=None)
    df = dc.convert(label_type=1, volume=True)

    # Build sliding windows
    X, y = build_windows(df, window_size=WINDOW_SIZE)

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=42, stratify=y
    )

    # Select prototypes from training set (positive class only)
    selector = PrototypeSelector(X_tr, y_tr, window_size=WINDOW_SIZE)
    protos, proto_labels, _, _ = selector.select_prototypes(
        num_prototypes=N_PROTOTYPES,
        selection_type='positive_only'
    )

    # Compute Euclidean distances from test windows to prototypes
    extractor = PrototypeFeatureExtractor(torch.from_numpy(X_te), torch.from_numpy(protos))
    dists = extractor._compute_euclidean_features().mean(dim=2).numpy()
    closest = np.argmin(dists, axis=1)

    # Summarize prototype influence
    counts = np.bincount(closest, minlength=N_PROTOTYPES)
    summary = pd.DataFrame({
        'prototype': np.arange(N_PROTOTYPES),
        'closest_count': counts,
        'closest_percentage': counts / len(closest)
    })
    summary_path = os.path.join('figures', 'prototype_influence_summary.csv')
    os.makedirs('figures', exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print('Prototype influence summary:')
    print(summary)

    # Visualization for a single test sample
    idx = 0
    nearest_idx = closest[idx]
    plt.figure(figsize=(8, 4))
    plt.plot(X_te[idx, :, 0], label='Test Close')
    plt.plot(protos[nearest_idx, :, 0], '--', label=f'Prototype {nearest_idx} Close')
    plt.axvline(WINDOW_SIZE // 2, color='red', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'test_case_vs_prototype.png'))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(N_PROTOTYPES), dists[idx])
    plt.xlabel('Prototype')
    plt.ylabel('Euclidean distance')
    plt.title(f'Distance for sample {idx}')
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'test_case_distance.png'))
    plt.close()


if __name__ == '__main__':
    main()
