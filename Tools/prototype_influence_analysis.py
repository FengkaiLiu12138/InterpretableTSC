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
N_PROTOTYPES = 10
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

    # Select prototypes from the training set
    selector = PrototypeSelector(X_tr, y_tr, window_size=WINDOW_SIZE)
    protos, proto_labels, _, _ = selector.select_prototypes(
        num_prototypes=N_PROTOTYPES,
        selection_type='random'
    )

    # Compute Euclidean distance features for train and test sets
    extractor_tr = PrototypeFeatureExtractor(torch.from_numpy(X_tr), torch.from_numpy(protos))
    feats_tr = extractor_tr._compute_euclidean_features().mean(dim=2).numpy()
    extractor_te = PrototypeFeatureExtractor(torch.from_numpy(X_te), torch.from_numpy(protos))
    feats_te = extractor_te._compute_euclidean_features().mean(dim=2).numpy()

    # Fit a simple logistic regression classifier on prototype distances
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000)
    clf.fit(feats_tr, y_tr)
    preds = clf.predict(feats_te)
    prob_pos = clf.predict_proba(feats_te)[:, 1]

    # Map each test sample to the closest prototype (for summary only)
    dists = feats_te
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

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_te, preds)
    print(f'Test Accuracy: {acc:.4f}')

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

    # Visualization of prototype contributions using arrows
    contrib = clf.coef_[0] * feats_te[idx]
    y_pos = np.arange(N_PROTOTYPES)
    colors = ['green' if c >= 0 else 'red' for c in contrib]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvline(0, color='black', linewidth=0.5)
    for i, val in enumerate(contrib):
        ax.annotate(
            '',
            xy=(val, i),
            xytext=(0, i),
            arrowprops=dict(arrowstyle='->', color=colors[i], lw=2),
        )
        ax.text(
            val + (0.02 if val >= 0 else -0.02),
            i,
            f'{val:.2f}',
            va='center',
            ha='left' if val >= 0 else 'right',
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'P{i}' for i in range(N_PROTOTYPES)])
    ax.set_xlabel('Contribution to logit')
    ax.set_title(
        f'Sample {idx} Prototype Influence\nPred prob={prob_pos[idx]:.2f} -> Class {preds[idx]}'
    )
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'sample_contribution_arrows.png'))
    plt.close()


if __name__ == '__main__':
    main()
