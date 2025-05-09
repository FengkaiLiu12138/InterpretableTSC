import torch
from torch import nn
from fastdtw import fastdtw


class PrototypeFeatureExtractor:
    def __init__(self, time_series, prototypes):
        """
        Args:
            data (torch.Tensor): 输入数据，形状为 (B, T, C)
            prototypes (torch.Tensor): 原型，形状为 (num_prototypes, C)
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
        # for each featuren in each time series, computer one euclidean distance to each prototype
        # return shape (B, num_prototypes)
        B, T, C = self.time_series.shape
        num_prototypes = self.prototypes.shape[0]
        features = torch.zeros(B, num_prototypes, C)
        for i in range(B):
            for j in range(num_prototypes):
                for k in range(C):
                    features[i, j, k] = torch.norm(self.time_series[i, :, k] - self.prototypes[j, k])
        return features

    def _compute_dtw_features(self):
        # for each featuren in each time series, computer one dtw distance to each prototype
        # return shape (B, num_prototypes)
        B, T, C = self.time_series.shape
        num_prototypes = self.prototypes.shape[0]
        features = torch.zeros(B, num_prototypes, C)
        for i in range(B):
            for j in range(num_prototypes):
                for k in range(C):
                    dist, _ = fastdtw(self.time_series[i, :, k].numpy(), self.prototypes[j, k].numpy())
                    features[i, j, k] = dist
        return features

    def _compute_cosine_features(self):
        # for each featuren in each time series, computer one cosine distance to each prototype
        # return shape (B, num_prototypes)
        B, T, C = self.time_series.shape
        num_prototypes = self.prototypes.shape[0]
        features = torch.zeros(B, num_prototypes, C)
        for i in range(B):
            for j in range(num_prototypes):
                for k in range(C):
                    features[i, j, k] = torch.nn.functional.cosine_similarity(self.time_series[i, :, k],
                                                                              self.prototypes[j, k])
        return features

    def _compute_mahalanobis_features(self):
        # for each featuren in each time series, computer one mahalanobis distance to each prototype
        # return shape (B, num_prototypes)
        B, T, C = self.time_series.shape
        num_prototypes = self.prototypes.shape[0]
        features = torch.zeros(B, num_prototypes, C)
        for i in range(B):
            for j in range(num_prototypes):
                for k in range(C):
                    diff = self.time_series[i, :, k] - self.prototypes[j, k]
                    cov_inv = torch.linalg.inv(torch.cov(self.time_series[i, :, k].numpy()))
                    features[i, j, k] = torch.sqrt(torch.matmul(torch.matmul(diff.T, cov_inv), diff))
        return features


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
        #  1) 将输入从 (B, P, C) 转置到 (B, C, P), 方便 conv1d 使用
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
        # residual 与 shortcut2 相加
        x = x + self.shortcut2(identity)

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
