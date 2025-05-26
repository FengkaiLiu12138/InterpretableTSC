import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastdtw import fastdtw


class PrototypeSelector:
    def __init__(self, data, labels, window_size=600):
        """
        data: shape (N, window_size, num_features) or (N, some_feature_dim)
        labels: shape (N,), binary (0 or 1)
        window_size: default 600
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.half_window = window_size // 2

    def select_prototypes(self, num_prototypes, selection_type='random'):
        """
        Select prototypes from the dataset based on the specified method and remove them from the dataset.

        Args
        ----
        num_prototypes : int
            Number of prototypes to select.
        selection_type : str
            Prototype selection strategy. Options are
            ``'random'`` (default), ``'positive'``/``'pos-only'``/``'positive_only'``
            for drawing only from the positive class, ``'k-means'`` and ``'gmm'``.

        Returns:
            prototypes (ndarray): shape (num_prototypes, ...)
            prototype_labels (ndarray): shape (num_prototypes,)
            remaining_data (ndarray): shape (N - num_prototypes, ...)
            remaining_labels (ndarray): shape (N - num_prototypes,)
        """
        if selection_type == 'random':
            return self.random_selection(num_prototypes)
        elif selection_type in ['positive', 'pos-only', 'positive_only']:
            return self.random_selection(num_prototypes, positive_only=True)
        elif selection_type == 'k-means':
            return self.k_means_selection(num_prototypes)
        elif selection_type == 'gmm':
            return self.gmm_selection(num_prototypes)
        else:
            raise ValueError(f"Unsupported selection type: {selection_type}")

    def random_selection(self, num_prototypes, positive_only: bool = False):
        """
        Randomly select prototypes from the dataset.

        If ``positive_only`` is ``True`` all prototypes are drawn from the
        positive class. Otherwise the method tries to select a roughly equal
        number of positive and negative samples when possible.

        Returns
        -------
        prototypes : ndarray
            Selected prototype samples.
        prototype_labels : ndarray
            Labels corresponding to ``prototypes``.
        remaining_data : ndarray
            Data that was not selected as prototypes.
        remaining_labels : ndarray
            Labels for ``remaining_data``.
        """
        pos_idx = np.where(self.labels == 1)[0]
        neg_idx = np.where(self.labels == 0)[0]
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        if positive_only:
            if len(pos_idx) >= num_prototypes:
                selected_idx = pos_idx[:num_prototypes]
            else:
                selected_idx = pos_idx.copy()
                extra = np.random.choice(pos_idx, num_prototypes - len(pos_idx), replace=True)
                selected_idx = np.concatenate([selected_idx, extra])
        else:
            half = num_prototypes // 2
            num_pos = min(half, len(pos_idx))
            num_neg = num_prototypes - num_pos
            num_neg = min(num_neg, len(neg_idx))

            selected_pos = pos_idx[:num_pos]
            selected_neg = neg_idx[:num_neg]
            selected_idx = np.concatenate([selected_pos, selected_neg])

            remainder = num_prototypes - len(selected_idx)
            if remainder > 0:
                leftover_idx = np.concatenate([pos_idx[num_pos:], neg_idx[num_neg:]])
                np.random.shuffle(leftover_idx)
                selected_idx = np.concatenate([selected_idx, leftover_idx[:remainder]])

        prototypes = self.data[selected_idx]
        prototype_labels = self.labels[selected_idx]

        mask = np.ones(len(self.data), dtype=bool)
        mask[selected_idx] = False
        remaining_data = self.data[mask]
        remaining_labels = self.labels[mask]

        return prototypes, prototype_labels, remaining_data, remaining_labels

    def k_means_selection(self, num_prototypes):
        """
        K-means clustering to select prototypes from the dataset. We cluster
        the entire dataset into 'num_prototypes' clusters. We then pick one sample
        (the closest to the cluster center) from each cluster as a prototype.

        Returns:
            prototypes, prototype_labels, remaining_data, remaining_labels
        """
        N = len(self.data)
        if self.data.ndim == 3:
            flat_data = self.data.reshape(N, -1)
        else:
            flat_data = self.data

        kmeans = KMeans(n_clusters=num_prototypes, random_state=42)
        kmeans.fit(flat_data)
        centers = kmeans.cluster_centers_
        labels_km = kmeans.labels_

        prototypes = []
        prototype_labels = []
        chosen_indices = []

        for c_idx in range(num_prototypes):
            cluster_indices = np.where(labels_km == c_idx)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_data = flat_data[cluster_indices]
            center = centers[c_idx]
            dists = np.sum((cluster_data - center) ** 2, axis=1)
            min_idx = np.argmin(dists)
            best_sample_global_idx = cluster_indices[min_idx]

            prototypes.append(self.data[best_sample_global_idx])
            prototype_labels.append(self.labels[best_sample_global_idx])
            chosen_indices.append(best_sample_global_idx)

        chosen_indices = np.array(chosen_indices, dtype=int)
        prototypes = np.array(prototypes)
        prototype_labels = np.array(prototype_labels)

        mask = np.ones(len(self.data), dtype=bool)
        mask[chosen_indices] = False
        remaining_data = self.data[mask]
        remaining_labels = self.labels[mask]

        return prototypes, prototype_labels, remaining_data, remaining_labels

    def gmm_selection(self, num_prototypes):
        """
        Gaussian mixture model to select prototypes from the dataset. Similar to K-means:
          - Fit GMM with 'num_prototypes' components
          - For each component, pick the sample closest to the component mean
            as the prototype.
        Returns:
            prototypes, prototype_labels, remaining_data, remaining_labels
        """
        N = len(self.data)
        if self.data.ndim == 3:
            flat_data = self.data.reshape(N, -1)
        else:
            flat_data = self.data

        gmm = GaussianMixture(n_components=num_prototypes, random_state=42)
        gmm.fit(flat_data)
        means = gmm.means_

        prototypes = []
        prototype_labels = []
        chosen_indices = []

        for i in range(num_prototypes):
            mean_i = means[i]
            dists = np.sum((flat_data - mean_i) ** 2, axis=1)
            min_idx = np.argmin(dists)
            prototypes.append(self.data[min_idx])
            prototype_labels.append(self.labels[min_idx])
            chosen_indices.append(min_idx)

        chosen_indices = np.array(chosen_indices, dtype=int)
        prototypes = np.array(prototypes)
        prototype_labels = np.array(prototype_labels)

        mask = np.ones(len(self.data), dtype=bool)
        mask[chosen_indices] = False
        remaining_data = self.data[mask]
        remaining_labels = self.labels[mask]

        return prototypes, prototype_labels, remaining_data, remaining_labels


###############################################################################
# 2) PrototypeFeatureExtractor
###############################################################################
class PrototypeFeatureExtractor:
    def __init__(self, time_series, prototypes):
        """
        Args:
            time_series (torch.Tensor): input time series, shape (B, T, C)
            prototypes (torch.Tensor): prototype, shape (num_prototypes, T, C)
        """
        self.time_series = time_series
        self.prototypes = prototypes

    def compute_prototype_features(self, metric='euclidean'):
        """
        calculate the distance between time series and prototypes
        Args:
            metric (str): 'euclidean' / 'dtw' / 'cosine'
        Returns:
            features (torch.Tensor): shape (B, num_prototypes, C)
        """
        if metric == 'euclidean':
            return self._compute_euclidean_features()
        elif metric == 'dtw':
            return self._compute_dtw_features()
        elif metric == 'cosine':
            return self._compute_cosine_features()
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def plot_prototype_feature_map(self, metric='euclidean', save_path="prototype_feature_map.png", sample_idx: int = 0):
        """Visualize a prototype feature map for a specific sample.

        Parameters
        ----------
        metric : str
            Distance metric used to compute the feature map.
        save_path : str
            Where to save the plotted heatmap.
        sample_idx : int
            Index of the sample to visualize. Defaults to ``0``.
        """
        features = self.compute_prototype_features(metric=metric)
        if features.shape[0] == 0 or sample_idx >= features.shape[0]:
            return
        sample_feat = features[sample_idx].cpu().numpy()

        plt.figure(figsize=(6, 4))
        sns.heatmap(sample_feat, annot=False, cmap="viridis")
        plt.title(f"Prototype Feature Map ({metric})")
        plt.xlabel("Feature Dimension")
        plt.ylabel("Prototype Index")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_prototype_cycles(
        self,
        short_window: int = 30,
        long_window: int = 120,
        save_dir: str = '.',
        prefix: str = 'prototype_cycle'
    ):
        """Plot each prototype with short/long moving averages.

        This visualization helps relate prototypes to long/short cycles,
        offering interpretable turning-point patterns.
        """
        os.makedirs(save_dir, exist_ok=True)
        num_prototypes = self.prototypes.shape[0]
        for idx in range(num_prototypes):
            proto = self.prototypes[idx]
            if isinstance(proto, torch.Tensor):
                proto_np = proto.cpu().numpy()
            else:
                proto_np = proto

            close_series = proto_np[:, 0]  # assume first channel is Close price
            short_ma = pd.Series(close_series).rolling(window=short_window).mean()
            long_ma = pd.Series(close_series).rolling(window=long_window).mean()

            plt.figure(figsize=(8, 4))
            plt.plot(close_series, label='Close', color='black')
            plt.plot(short_ma, label=f'MA{short_window}', linestyle='--')
            plt.plot(long_ma, label=f'MA{long_window}', linestyle='-.')
            plt.axvline(len(close_series)//2, color='red', linestyle=':', label='center')
            plt.title(f'Prototype {idx}')
            plt.xlabel('Time Step')
            plt.ylabel('Normalized Price')
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'{prefix}_{idx}.png')
            plt.savefig(save_path)
            plt.close()

    def plot_prototype_series(
        self,
        save_dir: str = '.',
        prefix: str = 'prototype_raw'
    ):
        """Plot each prototype's raw time series.

        Parameters
        ----------
        save_dir : str
            Directory to save the plots.
        prefix : str
            Prefix for the saved file names.
        """
        os.makedirs(save_dir, exist_ok=True)
        num_prototypes = self.prototypes.shape[0]
        for idx in range(num_prototypes):
            proto = self.prototypes[idx]
            if isinstance(proto, torch.Tensor):
                proto_np = proto.cpu().numpy()
            else:
                proto_np = proto

            plt.figure(figsize=(8, 4))
            for c in range(proto_np.shape[1]):
                plt.plot(proto_np[:, c], label=f'Var{c}')
            plt.axvline(proto_np.shape[0] // 2, color='red', linestyle=':', label='center')
            plt.title(f'Prototype {idx}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'{prefix}_{idx}.png')
            plt.savefig(save_path)
            plt.close()

    def _compute_euclidean_features(self):
        B, T, C = self.time_series.shape
        num_prototypes = self.prototypes.shape[0]
        features = torch.zeros(B, num_prototypes, C)
        for i in range(B):
            for j in range(num_prototypes):
                for k in range(C):
                    features[i, j, k] = torch.norm(
                        self.time_series[i, :, k] - self.prototypes[j, :, k]
                    )
        return features

    def _compute_dtw_features(self):
        B, T, C = self.time_series.shape
        num_prototypes = self.prototypes.shape[0]
        features = torch.zeros(B, num_prototypes, C)
        for i in range(B):
            for j in range(num_prototypes):
                for k in range(C):
                    dist, _ = fastdtw(
                        self.time_series[i, :, k].numpy(),
                        self.prototypes[j, :, k].numpy()
                    )
                    features[i, j, k] = dist
        return features

    def _compute_cosine_features(self):
        B, T, C = self.time_series.shape
        num_prototypes = self.prototypes.shape[0]
        features = torch.zeros(B, num_prototypes, C)
        for i in range(B):
            for j in range(num_prototypes):
                for k in range(C):
                    features[i, j, k] = F.cosine_similarity(
                        self.time_series[i, :, k],
                        self.prototypes[j, :, k],
                        dim=0
                    )
        return features


###############################################################################
# 3) PrototypeBasedModel
###############################################################################
class PrototypeModelBase(nn.Module):
    """Base class for all prototype-based models."""

    pass


###############################################################################
# 3) Prototype Models
###############################################################################

class PrototypeBasedModel(PrototypeModelBase):
    """
    Prototype-based model for time series classification using a ResNet
    backbone. This was the original prototype-based model in the project.
    """

    def __init__(self, num_prototypes: int, n_var: int, num_classes: int):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv1d(in_channels=n_var, out_channels=64,
                               kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        # Block 1 (64 filters)
        self.block1_conv1 = nn.Conv1d(64, 64, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block1_bn1 = nn.BatchNorm1d(64)
        self.block1_conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block1_bn2 = nn.BatchNorm1d(64)
        self.block1_conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block1_bn3 = nn.BatchNorm1d(64)

        # Block 2 (128 filters)
        self.block2_conv1 = nn.Conv1d(64, 128, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block2_bn1 = nn.BatchNorm1d(128)
        self.block2_conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block2_bn2 = nn.BatchNorm1d(128)
        self.block2_conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block2_bn3 = nn.BatchNorm1d(128)
        self.shortcut2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128)
        )

        # Block 3 (128 filters)
        self.block3_conv1 = nn.Conv1d(128, 128, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block3_bn1 = nn.BatchNorm1d(128)
        self.block3_conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block3_bn2 = nn.BatchNorm1d(128)
        self.block3_conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block3_bn3 = nn.BatchNorm1d(128)

        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_prototypes, n_var)
        Returns:
            logits: (B, num_classes)
        """
        x = x.transpose(1, 2)  # (B, n_var, num_prototypes)

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Block 1
        identity = x
        x = self.relu(self.block1_bn1(self.block1_conv1(x)))
        x = self.relu(self.block1_bn2(self.block1_conv2(x)))
        x = self.relu(self.block1_bn3(self.block1_conv3(x)))
        x = x + identity

        # Block 2
        identity = x
        x = self.relu(self.block2_bn1(self.block2_conv1(x)))
        x = self.relu(self.block2_bn2(self.block2_conv2(x)))
        x = self.relu(self.block2_bn3(self.block2_conv3(x)))
        x = x + self.shortcut2(identity)

        # Block 3
        identity = x
        x = self.relu(self.block3_bn1(self.block3_conv1(x)))
        x = self.relu(self.block3_bn2(self.block3_conv2(x)))
        x = self.relu(self.block3_bn3(self.block3_conv3(x)))
        x = x + identity

        x = self.global_pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits

    def forward_with_intermediate(self, x: torch.Tensor):
        """
        get intermediate features
        Args:
            x: (B, num_prototypes, n_var)
        Returns:
            outs: list of intermediate features
        """
        outs = []
        x = x.transpose(1, 2)  # (B, n_var, num_prototypes)

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Block 1
        identity = x
        x = self.relu(self.block1_bn1(self.block1_conv1(x)))
        x = self.relu(self.block1_bn2(self.block1_conv2(x)))
        x = self.relu(self.block1_bn3(self.block1_conv3(x)))
        x = x + identity
        outs.append(x.clone().detach())

        # Block 2
        identity = x
        x = self.relu(self.block2_bn1(self.block2_conv1(x)))
        x = self.relu(self.block2_bn2(self.block2_conv2(x)))
        x = self.relu(self.block2_bn3(self.block2_conv3(x)))
        x = x + self.shortcut2(identity)
        outs.append(x.clone().detach())

        # Block 3
        identity = x
        x = self.relu(self.block3_bn1(self.block3_conv1(x)))
        x = self.relu(self.block3_bn2(self.block3_conv2(x)))
        x = self.relu(self.block3_bn3(self.block3_conv3(x)))
        x = x + identity
        outs.append(x.clone().detach())

        return outs


###############################################################################
# 4) Wrapper classes for baseline backbones
###############################################################################

class PrototypeCNN(PrototypeModelBase):
    """Prototype model using the CNN baseline architecture."""

    def __init__(self, num_prototypes: int, n_var: int, num_classes: int):
        super().__init__()
        from BaselineModel.CNN_baseline import CNN

        self.backbone = CNN(num_prototypes, n_var, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class PrototypeFCN(PrototypeModelBase):
    """Prototype model using the FCN baseline architecture."""

    def __init__(self, num_prototypes: int, n_var: int, num_classes: int):
        super().__init__()
        from BaselineModel.FCN_baseline import FCN

        self.backbone = FCN(num_prototypes, n_var, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class PrototypeLSTM(PrototypeModelBase):
    """Prototype model using the LSTM baseline architecture."""

    def __init__(self, num_prototypes: int, n_var: int, num_classes: int,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        from BaselineModel.LSTM_baseline import LSTM

        self.backbone = LSTM(
            window_size=num_prototypes,
            n_vars=n_var,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class PrototypeMLP(PrototypeModelBase):
    """Prototype model using the MLP baseline architecture."""

    def __init__(self, num_prototypes: int, n_var: int, num_classes: int):
        super().__init__()
        from BaselineModel.MLP_baseline import MLP

        self.backbone = MLP(num_prototypes, n_var, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class PrototypeResNet(PrototypeModelBase):
    """Prototype model using the ResNet baseline architecture."""

    def __init__(self, num_prototypes: int, n_var: int, num_classes: int):
        super().__init__()
        from BaselineModel.ResNet_baseline import ResNet

        self.backbone = ResNet(num_prototypes, n_var, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
