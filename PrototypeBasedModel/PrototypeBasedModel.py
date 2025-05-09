import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F

# 如果要使用 DTW，需要先 pip install fastdtw
from fastdtw import fastdtw


###############################################################################
# 1) PrototypeSelector
###############################################################################
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

        Args:
            num_prototypes (int): number of prototypes to select
            selection_type (str): 'random' / 'k-means' / 'gmm'

        Returns:
            prototypes (ndarray): shape (num_prototypes, ...)
            prototype_labels (ndarray): shape (num_prototypes,)
            remaining_data (ndarray): shape (N - num_prototypes, ...)
            remaining_labels (ndarray): shape (N - num_prototypes,)
        """
        if selection_type == 'random':
            return self.random_selection(num_prototypes)
        elif selection_type == 'k-means':
            return self.k_means_selection(num_prototypes)
        elif selection_type == 'gmm':
            return self.gmm_selection(num_prototypes)
        else:
            raise ValueError(f"Unsupported selection type: {selection_type}")

    def random_selection(self, num_prototypes):
        """
        Randomly select prototypes from the dataset, trying to get a roughly equal
        number of positive and negative samples (if possible).

        Returns:
            prototypes, prototype_labels, remaining_data, remaining_labels
        """
        # Separate indices for pos & neg
        pos_idx = np.where(self.labels == 1)[0]
        neg_idx = np.where(self.labels == 0)[0]

        # Shuffle
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        # Half from pos, half from neg (if possible)
        half = num_prototypes // 2
        num_pos = min(half, len(pos_idx))
        num_neg = num_prototypes - num_pos
        num_neg = min(num_neg, len(neg_idx))

        selected_pos = pos_idx[:num_pos]
        selected_neg = neg_idx[:num_neg]
        selected_idx = np.concatenate([selected_pos, selected_neg])

        # If we still haven't reached num_prototypes (e.g. minority class is too small)
        # We can randomly pick the remainder from whichever class has leftover.
        remainder = num_prototypes - len(selected_idx)
        if remainder > 0:
            # Combine leftover indices
            leftover_idx = np.concatenate([pos_idx[num_pos:], neg_idx[num_neg:]])
            np.random.shuffle(leftover_idx)
            selected_idx = np.concatenate([selected_idx, leftover_idx[:remainder]])

        # Extract prototypes
        prototypes = self.data[selected_idx]
        prototype_labels = self.labels[selected_idx]

        # Build remaining
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
        # Flatten each sample if needed. Example shape: (N, window_size*features)
        N = len(self.data)
        if self.data.ndim == 3:
            # (N, window_size, feature_dim) -> (N, window_size*feature_dim)
            flat_data = self.data.reshape(N, -1)
        else:
            flat_data = self.data

        kmeans = KMeans(n_clusters=num_prototypes, random_state=42)
        kmeans.fit(flat_data)
        centers = kmeans.cluster_centers_
        labels_km = kmeans.labels_  # cluster assignment for each sample

        prototypes = []
        prototype_labels = []
        chosen_indices = []

        for c_idx in range(num_prototypes):
            # samples in cluster c_idx
            cluster_indices = np.where(labels_km == c_idx)[0]
            if len(cluster_indices) == 0:
                # If a cluster is empty (rare but can happen), skip
                continue

            # find the sample closest to the center
            cluster_data = flat_data[cluster_indices]
            center = centers[c_idx]

            # dist^2 from center
            dists = np.sum((cluster_data - center) ** 2, axis=1)
            min_idx = np.argmin(dists)
            best_sample_global_idx = cluster_indices[min_idx]

            prototypes.append(self.data[best_sample_global_idx])
            prototype_labels.append(self.labels[best_sample_global_idx])
            chosen_indices.append(best_sample_global_idx)

        chosen_indices = np.array(chosen_indices, dtype=int)
        prototypes = np.array(prototypes)
        prototype_labels = np.array(prototype_labels)

        # Build remaining
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
        # Flatten each sample if needed.
        N = len(self.data)
        if self.data.ndim == 3:
            flat_data = self.data.reshape(N, -1)
        else:
            flat_data = self.data

        gmm = GaussianMixture(n_components=num_prototypes, random_state=42)
        gmm.fit(flat_data)

        # Means of shape (num_prototypes, feature_dim)
        means = gmm.means_

        prototypes = []
        prototype_labels = []
        chosen_indices = []

        for i in range(num_prototypes):
            mean_i = means[i]
            # find the sample closest to this component mean
            dists = np.sum((flat_data - mean_i) ** 2, axis=1)
            min_idx = np.argmin(dists)
            prototypes.append(self.data[min_idx])
            prototype_labels.append(self.labels[min_idx])
            chosen_indices.append(min_idx)

        chosen_indices = np.array(chosen_indices, dtype=int)
        prototypes = np.array(prototypes)
        prototype_labels = np.array(prototype_labels)

        # Build remaining
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
            time_series (torch.Tensor): 输入数据，形状为 (B, T, C)
            prototypes (torch.Tensor): 原型，形状为 (num_prototypes, T, C)
        """
        self.time_series = time_series
        self.prototypes = prototypes

    def compute_prototype_features(self, metric='euclidean'):
        """
        计算原型特征
        Args:
            metric (str): 距离度量方式，默认使用欧氏距离
            - 可选 'euclidean', 'dtw', 'cosine', 'mahalanobis'
        Returns:
            torch.Tensor: 原型特征，形状为 (B, num_prototypes, C)
        """
        # 计算原型特征
        if metric == 'euclidean':
            return self._compute_euclidean_features()
        elif metric == 'dtw':
            # 计算动态时间规整距离 (DTW)
            return self._compute_dtw_features()
        elif metric == 'cosine':
            # 计算余弦相似度
            return self._compute_cosine_features()
        elif metric == 'mahalanobis':
            # 计算马氏距离
            return self._compute_mahalanobis_features()
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def plot_prototype_feature_map(self, metric='euclidean'):
        pass

    def _compute_euclidean_features(self):
        # for each feature in each time series, compute one euclidean distance to each prototype
        # return shape (B, num_prototypes, C)
        B, T, C = self.time_series.shape
        num_prototypes = self.prototypes.shape[0]
        features = torch.zeros(B, num_prototypes, C)
        for i in range(B):
            for j in range(num_prototypes):
                for k in range(C):
                    # 修复：改成 self.prototypes[j, :, k]
                    features[i, j, k] = torch.norm(
                        self.time_series[i, :, k] - self.prototypes[j, :, k]
                    )
        return features

    def _compute_dtw_features(self):
        # for each feature in each time series, compute one DTW distance to each prototype
        # return shape (B, num_prototypes, C)
        B, T, C = self.time_series.shape
        num_prototypes = self.prototypes.shape[0]
        features = torch.zeros(B, num_prototypes, C)
        for i in range(B):
            for j in range(num_prototypes):
                for k in range(C):
                    # 修复：改成 self.prototypes[j, :, k]
                    dist, _ = fastdtw(
                        self.time_series[i, :, k].numpy(),
                        self.prototypes[j, :, k].numpy()
                    )
                    features[i, j, k] = dist
        return features

    def _compute_cosine_features(self):
        # for each feature in each time series, compute one cosine similarity to each prototype
        # return shape (B, num_prototypes, C)
        B, T, C = self.time_series.shape
        num_prototypes = self.prototypes.shape[0]
        features = torch.zeros(B, num_prototypes, C)
        for i in range(B):
            for j in range(num_prototypes):
                for k in range(C):
                    # 修复：改成 self.prototypes[j, :, k]
                    features[i, j, k] = F.cosine_similarity(
                        self.time_series[i, :, k],
                        self.prototypes[j, :, k],
                        dim=0
                    )
        return features

    def _compute_mahalanobis_features(self):
        # for each feature in each time series, compute one mahalanobis distance to each prototype
        # return shape (B, num_prototypes, C)
        B, T, C = self.time_series.shape
        num_prototypes = self.prototypes.shape[0]
        features = torch.zeros(B, num_prototypes, C)
        for i in range(B):
            for j in range(num_prototypes):
                for k in range(C):
                    # 修复：改成 self.prototypes[j, :, k]
                    diff = self.time_series[i, :, k] - self.prototypes[j, :, k]
                    cov_inv = torch.linalg.inv(torch.cov(self.time_series[i, :, k].numpy()))
                    features[i, j, k] = torch.sqrt(torch.matmul(torch.matmul(diff.T, cov_inv), diff))
        return features


###############################################################################
# 3) PrototypeBasedModel
###############################################################################

class PrototypeBasedModel(nn.Module):
    """
    一个示例：与最初的 ResNet 架构类似，只是输入的形状换成 (B, num_prototypes, C)。
    """

    def __init__(self, num_prototypes: int, n_var: int, num_classes: int):
        super().__init__()

        # Stem: 第一个卷积，输入通道= n_var
        self.conv1 = nn.Conv1d(in_channels=n_var, out_channels=64,
                               kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        # ---------- Block 1 (64 filters) ----------
        self.block1_conv1 = nn.Conv1d(64, 64, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block1_bn1 = nn.BatchNorm1d(64)
        self.block1_conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block1_bn2 = nn.BatchNorm1d(64)
        self.block1_conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block1_bn3 = nn.BatchNorm1d(64)
        # 无 shortcut，因为输入输出通道数相同

        # ---------- Block 2 (128 filters) ----------
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

        # ---------- Block 3 (128 filters) ----------
        self.block3_conv1 = nn.Conv1d(128, 128, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block3_bn1 = nn.BatchNorm1d(128)
        self.block3_conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block3_bn2 = nn.BatchNorm1d(128)
        self.block3_conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1,
                                      padding=1, bias=False)
        self.block3_bn3 = nn.BatchNorm1d(128)
        # 这里不需要额外的 shortcut，通道保持不变

        # 全局池化 + 全连接层输出
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)  # (B, 128, 1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_prototypes, n_var)
        Returns:
            logits: (B, num_classes)
        """
        # --------------------------------------------------------
        #  将输入从 (B, P, C) 转置到 (B, C, P), 方便 conv1d 使用
        # --------------------------------------------------------
        x = x.transpose(1, 2)  # (B, C, P)

        # --------------------- Stem ---------------------
        x = self.conv1(x)  # (B, 64, P)
        x = self.bn1(x)
        x = self.relu(x)

        # ----------------- Block 1 ------------------
        identity = x
        x = self.relu(self.block1_bn1(self.block1_conv1(x)))
        x = self.relu(self.block1_bn2(self.block1_conv2(x)))
        x = self.relu(self.block1_bn3(self.block1_conv3(x)))
        x = x + identity  # 残差连接

        # ----------------- Block 2 ------------------
        identity = x
        x = self.relu(self.block2_bn1(self.block2_conv1(x)))
        x = self.relu(self.block2_bn2(self.block2_conv2(x)))
        x = self.relu(self.block2_bn3(self.block2_conv3(x)))
        x = x + self.shortcut2(identity)  # residual 与 shortcut2 相加

        # ----------------- Block 3 ------------------
        identity = x
        x = self.relu(self.block3_bn1(self.block3_conv1(x)))
        x = self.relu(self.block3_bn2(self.block3_conv2(x)))
        x = self.relu(self.block3_bn3(self.block3_conv3(x)))
        x = x + identity  # 残差连接

        # ---------------- Head: 全局池化 + FC ----------------
        x = self.global_pool(x).squeeze(-1)  # (B, 128)
        logits = self.fc(x)  # (B, num_classes)
        return logits

