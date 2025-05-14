import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Three-layer CNN model for sequence classification:
      Input shape: (B, T, C)
        1) Transpose to (B, C, T) to match Conv1d input format
        2) Pass through 3 convolutional blocks (Conv + BN + ReLU)
        3) Global average pooling
        4) Fully connected layer for classification
    """

    def __init__(self, window_size: int, n_vars: int, num_classes: int):
        """
        Args:
            window_size (int): Length of the time series T (can be stored or ignored as needed)
            n_vars (int): Number of features C
            num_classes (int): Number of classes
        """
        super().__init__()

        # 1st convolution: in_channels = n_vars, out_channels = 64
        self.conv1 = nn.Conv1d(in_channels=n_vars, out_channels=64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        # 2nd convolution: in_channels = 64, out_channels = 128
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # 3rd convolution: in_channels = 128, out_channels = 256
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()

        # Global average pooling (pool sequence dimension to 1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Final classification layer
        self.fc  = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (B, T, C)
          → First transposed to (B, C, T) for convolution
          → 3 Conv layers + ReLU
          → Global Average Pooling => (B, 256)
          → Fully connected => (B, num_classes)
        """
        # Transpose (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)

        # 1st convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 2nd convolution block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # 3rd convolution block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Global average pooling: (B, 256, T') -> (B, 256, 1) -> (B, 256)
        x = self.gap(x).squeeze(-1)

        # Classification output
        out = self.fc(x)
        return out
