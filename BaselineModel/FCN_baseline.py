import torch
import torch.nn as nn

class FCN(nn.Module):
    """
    FCN model with the structure:
    Input shape: (B, T, C)
       1) 先转置为 (B, C, T) 以适配 nn.Conv1d 的输入格式
       2) Conv1d(128) + BN + ReLU
       3) Conv1d(256) + BN + ReLU
       4) Conv1d(128) + BN + ReLU
       5) Global average pooling
       6) 全连接层输出至 num_classes

    说明:
      - T = window_size
      - C = n_vars (特征数, e.g. 5)
      - 若 kernel_size=5 并想要 "same" padding，需要手动指定 padding=2 (或使用 padding="same" 需 PyTorch>=2.0)
    """

    def __init__(self, n_vars: int, num_classes: int):
        """
        Args:
            n_vars (int): 每个时间步的特征数 (channels), 例如 5
            num_classes (int): 分类类别数
        """
        super().__init__()

        # 第一卷积：输入通道 = n_vars, 输出通道 = 128
        # kernel_size=8 不做 padding，相当于有效卷积
        self.conv1 = nn.Conv1d(in_channels=n_vars, out_channels=128, kernel_size=8)
        self.bn1   = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        # 第二卷积：输入通道 = 128, 输出通道 = 256
        # kernel_size=5 时可用 padding=2 模拟 "same" 卷积
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        # 第三卷积：输入通道 = 256, 输出通道 = 128
        # kernel_size=3 时可用 padding=1 模拟 "same"
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        # 全局平均池化到长度 1
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 最终全连接分类
        self.fc  = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, T, C)
          -> 先转置为 (B, C, T)
          -> 经过 3 个卷积块 + 全局平均池化
          -> 输出线性分类结果
        """
        # 转置为 (B, C, T)
        x = x.transpose(1, 2)

        # Conv Block #1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Conv Block #2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Conv Block #3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Global Average Pooling: (B, 128, T') -> (B, 128, 1) -> (B, 128)
        x = self.gap(x).squeeze(-1)

        # 输出分类: (B, num_classes)
        x = self.fc(x)
        return x
