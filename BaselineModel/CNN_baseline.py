import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    三层 CNN 模型，用于时序分类:
      输入形状: (B, T, C)
        1) 先转置为 (B, C, T)，以适配 Conv1d 的输入格式
        2) 依次通过 3 个卷积块 (Conv + BN + ReLU)
        3) 全局平均池化
        4) 全连接输出分类
    """

    def __init__(self, window_size: int, n_vars: int, num_classes: int):
        """
        Args:
            window_size (int): 时间序列长度 T (可根据需要使用, 也可不存储)
            n_vars (int): 特征数 C
            num_classes (int): 分类类别数
        """
        super().__init__()

        # 第1层卷积：输入通道 = n_vars, 输出通道 = 64
        self.conv1 = nn.Conv1d(in_channels=n_vars, out_channels=64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        # 第2层卷积：输入通道 = 64, 输出通道 = 128
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # 第3层卷积：输入通道 = 128, 输出通道 = 256
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()

        # 全局平均池化 (将序列维度池化成1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 最终分类
        self.fc  = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x 形状: (B, T, C)
          → 先转成 (B, C, T) 再卷积
          → 3 层 Conv + ReLU
          → Global Average Pooling => (B, 256)
          → 全连接 => (B, num_classes)
        """
        # 转置 (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)

        # 第1层卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 第2层卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # 第3层卷积
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # 全局平均池化: (B, 256, T') -> (B, 256, 1) -> (B, 256)
        x = self.gap(x).squeeze(-1)

        # 分类输出
        out = self.fc(x)
        return out