import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
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


def train_model(model, train_loader, epochs=3, lr=1e-3):
    """Simple training loop for baseline models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    return model


def gradient_importance(model, sample, class_idx=1):
    """Compute gradient-based importance for a single sample."""
    device = next(model.parameters()).device
    inp = torch.from_numpy(sample[np.newaxis]).to(device)
    inp.requires_grad_(True)
    out = model(inp)
    score = out[0, class_idx]
    model.zero_grad()
    score.backward()
    grad = inp.grad.abs().detach().cpu().numpy()[0]
    return grad.mean(axis=1)


def plot_prototype_influence(idx, X_te, protos, contrib, prob, pred):
    """Plot prototype contribution grid."""
    colors = ['green' if c >= 0 else 'red' for c in contrib]
    from matplotlib import gridspec

    cols = 2
    rows = int(np.ceil(N_PROTOTYPES / cols))

    fig = plt.figure(figsize=(cols * 6, rows * 2 + 3))
    gs = gridspec.GridSpec(rows + 1, cols, hspace=0.6, wspace=0.4)

    for i in range(N_PROTOTYPES):
        r, c = divmod(i, cols)
        ax_p = fig.add_subplot(gs[r, c])
        ax_p.plot(protos[i, :, 0], color='blue')
        ax_p.axvline(WINDOW_SIZE // 2, color='red', ls=':')
        ax_p.set_title(f'P{i}  [{contrib[i]:.2f}]', color=colors[i])
        ax_p.set_xticks([])
        ax_p.set_yticks([])

    ax_test = fig.add_subplot(gs[rows, :])
    ax_test.plot(X_te[idx, :, 0], color='black')
    ax_test.axvline(WINDOW_SIZE // 2, color='red', ls=':')
    ax_test.set_title(f'Test sample {idx}\nProb={prob:.2f} -> Class {pred}')
    ax_test.set_xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'sample_contribution_grid.png'))
    plt.close()


def plot_gradient(sample, grad, fname, title):
    """Plot input signal with gradient-based importance overlay."""
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(sample[:, 0], color='black', label='Close')
    plt.fill_between(np.arange(len(grad)), sample[:, 0], sample[:, 0] + grad, color='red', alpha=0.3, label='Grad |dL/dx|')
    plt.axvline(WINDOW_SIZE // 2, color='red', ls=':')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('figures', fname))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Prototype influence and model comparison')
    parser.add_argument('--model', choices=['logreg', 'fcn', 'mlp'], default='logreg', help='Model for additional visualisation')
    args = parser.parse_args()

    # Load and convert dataset
    dc = DatasetConverter(file_path=DATA_PATH, save_path=None)
    df = dc.convert(label_type=1, volume=True)

    # Build sliding windows
    X, y = build_windows(df, window_size=WINDOW_SIZE)

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=42, stratify=y
    )

    # Select prototypes from the training set for the logreg baseline
    selector = PrototypeSelector(X_tr, y_tr, window_size=WINDOW_SIZE)
    protos, proto_labels, _, _ = selector.select_prototypes(
        num_prototypes=N_PROTOTYPES,
        selection_type='random'
    )

    # Compute Euclidean distance features
    extractor_tr = PrototypeFeatureExtractor(torch.from_numpy(X_tr), torch.from_numpy(protos))
    feats_tr = extractor_tr._compute_euclidean_features().mean(dim=2).numpy()
    extractor_te = PrototypeFeatureExtractor(torch.from_numpy(X_te), torch.from_numpy(protos))
    feats_te = extractor_te._compute_euclidean_features().mean(dim=2).numpy()

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(feats_tr, y_tr)
    preds = clf.predict(feats_te)
    prob_pos = clf.predict_proba(feats_te)[:, 1]

    # Map each test sample to the closest prototype
    closest = np.argmin(feats_te, axis=1)

    # Summarize prototype influence
    counts = np.bincount(closest, minlength=N_PROTOTYPES)
    summary = pd.DataFrame({
        'prototype': np.arange(N_PROTOTYPES),
        'closest_count': counts,
        'closest_percentage': counts / len(closest)
    })
    os.makedirs('figures', exist_ok=True)
    summary.to_csv(os.path.join('figures', 'prototype_influence_summary.csv'), index=False)
    print('Prototype influence summary:')
    print(summary)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_te, preds)
    print(f'LogReg Accuracy: {acc:.4f}')

    # Visualize contributions for one sample
    idx = 0
    contrib = clf.coef_[0] * feats_te[idx]
    plot_prototype_influence(idx, X_te, protos, contrib, prob_pos[idx], preds[idx])

    if args.model in {'fcn', 'mlp'}:
        # Prepare data loaders
        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64)

        if args.model == 'fcn':
            from BaselineModel.FCN_baseline import FCN
            model = FCN(window_size=WINDOW_SIZE, n_vars=X.shape[2], num_classes=2)
            name = 'FCN'
        else:
            from BaselineModel.MLP_baseline import MLP
            model = MLP(window_size=WINDOW_SIZE, n_vars=X.shape[2], num_classes=2)
            name = 'MLP'

        model = train_model(model, train_loader)

        # Evaluate accuracy
        model.eval()
        device = next(model.parameters()).device
        all_preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                out = model(xb)
                all_preds.append(out.argmax(dim=1).cpu())
        all_preds = torch.cat(all_preds).numpy()
        acc = accuracy_score(y_te, all_preds)
        print(f'{name} Accuracy: {acc:.4f}')

        grad = gradient_importance(model, X_te[idx])
        plot_gradient(X_te[idx], grad, f'{name.lower()}_gradient.png', f'{name} Gradient Importance')


if __name__ == '__main__':
    main()