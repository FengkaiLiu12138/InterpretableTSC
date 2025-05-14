import torch
import torch.nn as nn

class FCN(nn.Module):
    """
    FCN model with the structure:
    Input shape: (B, T, C)
       1) Transpose to (B, C, T) to match nn.Conv1d input format
       2) Conv1d(128) + BN + ReLU
       3) Conv1d(256) + BN + ReLU
       4) Conv1d(128) + BN + ReLU
       5) Global average pooling
       6) Fully connected layer to num_classes

    Notes:
      - T = window_size
      - C = n_vars (number of features, e.g. 5)
      - If kernel_size=5 and you want "same" padding, manually set padding=2
        (or use padding="same" with PyTorch ≥ 2.0)
    """

    def __init__(self, window_size: int, n_vars: int, num_classes: int):
        """
        Args:
            n_vars (int): Number of features (channels) at each time step, e.g. 5
            num_classes (int): Number of classes
        """
        super().__init__()

        # 1st convolution: in_channels = n_vars, out_channels = 128
        # No padding with kernel_size=8 (valid convolution)
        self.conv1 = nn.Conv1d(in_channels=n_vars, out_channels=128, kernel_size=8)
        self.bn1   = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        # 2nd convolution: in_channels = 128, out_channels = 256
        # Use padding=2 with kernel_size=5 to emulate "same" convolution
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        # 3rd convolution: in_channels = 256, out_channels = 128
        # Use padding=1 with kernel_size=3 to emulate "same"
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        # Global average pooling to length 1
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Final fully connected classifier
        self.fc  = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (B, T, C)
          -> First transpose to (B, C, T)
          -> Pass through 3 convolution blocks + global average pooling
          -> Output linear classification result
        """
        # Transpose to (B, C, T)
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

        # Classification output: (B, num_classes)
        x = self.fc(x)
        return x
