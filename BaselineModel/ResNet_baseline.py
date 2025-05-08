import torch
from torch import nn

class ResNet(nn.Module):
    """
    ResNet for 1‑D multivariate time series:
       Input shape: (B, T, C)  with T=window_size, C=n_vars=5
       Internally transposed to (B, C, T) for Conv1D.

    Args:
        window_size (int): Number of time steps in each sample.
        n_vars (int): Number of variables (features) per time step, e.g. 5.
        num_classes (int): Classification classes, e.g. 2 for binary.
    """
    def __init__(self, window_size: int, n_vars: int, num_classes: int):
        super().__init__()

        # The first conv uses `n_vars` as in_channels
        self.conv1 = nn.Conv1d(n_vars, 64, kernel_size=7, stride=1, padding=3, bias=False)

        # ---------- Block 1 (64 filters) ----------
        self.block1_conv1 = nn.Conv1d(64, 64, 3, 1, 1, bias=False)
        self.block1_bn1   = nn.BatchNorm1d(64)
        self.block1_conv2 = nn.Conv1d(64, 64, 3, 1, 1, bias=False)
        self.block1_bn2   = nn.BatchNorm1d(64)
        self.block1_conv3 = nn.Conv1d(64, 64, 3, 1, 1, bias=False)
        self.block1_bn3   = nn.BatchNorm1d(64)

        # ---------- Block 2 (128 filters) ----------
        self.block2_conv1 = nn.Conv1d(64, 128, 3, 1, 1, bias=False)
        self.block2_bn1   = nn.BatchNorm1d(128)
        self.block2_conv2 = nn.Conv1d(128, 128, 3, 1, 1, bias=False)
        self.block2_bn2   = nn.BatchNorm1d(128)
        self.block2_conv3 = nn.Conv1d(128, 128, 3, 1, 1, bias=False)
        self.block2_bn3   = nn.BatchNorm1d(128)
        self.shortcut2    = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128)
        )

        # ---------- Block 3 (128 filters) ----------
        self.block3_conv1 = nn.Conv1d(128, 128, 3, 1, 1, bias=False)
        self.block3_bn1   = nn.BatchNorm1d(128)
        self.block3_conv2 = nn.Conv1d(128, 128, 3, 1, 1, bias=False)
        self.block3_bn2   = nn.BatchNorm1d(128)
        self.block3_conv3 = nn.Conv1d(128, 128, 3, 1, 1, bias=False)
        self.block3_bn3   = nn.BatchNorm1d(128)

        # Global pooling + FC
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc          = nn.Linear(128, num_classes)
        self.relu        = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  →  (B, C, T)
        x = x.transpose(1, 2)

        # Stem
        x = self.conv1(x)

        # ---------- Block 1 ----------
        identity = x
        for conv, bn in [
            (self.block1_conv1, self.block1_bn1),
            (self.block1_conv2, self.block1_bn2),
            (self.block1_conv3, self.block1_bn3),
        ]:
            x = self.relu(bn(conv(x)))
        x = x + identity  # residual 1

        # ---------- Block 2 ----------
        identity = x
        for conv, bn in [
            (self.block2_conv1, self.block2_bn1),
            (self.block2_conv2, self.block2_bn2),
            (self.block2_conv3, self.block2_bn3),
        ]:
            x = self.relu(bn(conv(x)))
        x = x + self.shortcut2(identity)  # residual 2

        # ---------- Block 3 ----------
        identity = x
        for conv, bn in [
            (self.block3_conv1, self.block3_bn1),
            (self.block3_conv2, self.block3_bn2),
            (self.block3_conv3, self.block3_bn3),
        ]:
            x = self.relu(bn(conv(x)))
        x = x + identity  # residual 3

        # Head
        x = self.global_pool(x).squeeze(-1)  # shape (B, 128)
        return self.fc(x)
