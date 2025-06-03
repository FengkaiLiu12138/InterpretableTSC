import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Tools.DatasetConverter import DatasetConverter
from PrototypeBasedModel import PrototypeSelector, PrototypeFeatureExtractor

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'ftse_minute_data_daily.csv')
WINDOW_SIZE = 600
N_PROTOTYPES = 10


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
    X, y = build_windows(df, window_size=WINDOW_SIZE)

    # Select prototypes from positive class
    selector = PrototypeSelector(X, y, window_size=WINDOW_SIZE)
    protos, _, _, _ = selector.select_prototypes(
        num_prototypes=N_PROTOTYPES,
        selection_type='positive_only',
    )

    # Compute Euclidean distance features
    extractor = PrototypeFeatureExtractor(torch.from_numpy(X), torch.from_numpy(protos))
    features = extractor._compute_euclidean_features().mean(dim=2).numpy()

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(features)

    # Simple logistic regression using prototype features
    clf = LogisticRegression(max_iter=1000)
    clf.fit(feats_scaled, y)
    preds = clf.predict(feats_scaled)
    acc = accuracy_score(y, preds)
    print(f'LogReg accuracy (prototype features): {acc:.4f}')

    # Prototype features for prototypes themselves
    proto_extractor = PrototypeFeatureExtractor(torch.from_numpy(protos), torch.from_numpy(protos))
    proto_feats = proto_extractor._compute_euclidean_features().mean(dim=2).numpy()
    proto_feats_scaled = scaler.transform(proto_feats)

    # t-SNE embedding on combined data+prototypes
    combined = np.vstack([feats_scaled, proto_feats_scaled])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embed = tsne.fit_transform(combined)
    embed_data = embed[: len(X)]
    embed_proto = embed[len(X) :]

    os.makedirs('figures', exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(embed_data[y == 0, 0], embed_data[y == 0, 1], s=10, alpha=0.5, label='Class 0')
    plt.scatter(embed_data[y == 1, 0], embed_data[y == 1, 1], s=10, alpha=0.5, label='Class 1')
    plt.scatter(embed_proto[:, 0], embed_proto[:, 1], color='red', marker='*', s=100, label='Prototypes')
    plt.title('t-SNE of Prototype Feature Space')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'tsne_prototype_features.png'))
    plt.close()

    # Highlight misclassified samples
    mis_idx = np.where(preds != y)[0]
    if len(mis_idx) > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(embed_data[:, 0], embed_data[:, 1], s=10, c=y, cmap='coolwarm', alpha=0.3)
        plt.scatter(embed_data[mis_idx, 0], embed_data[mis_idx, 1], color='black', label='Misclassified', s=20)
        plt.scatter(embed_proto[:, 0], embed_proto[:, 1], color='red', marker='*', s=100, label='Prototypes')
        plt.title('Misclassified Samples in t-SNE Space')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join('figures', 'tsne_misclassified.png'))
        plt.close()


if __name__ == '__main__':
    main()
