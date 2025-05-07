import torch.nn as nn


class FCN(nn.Module):
    """
    FCN model with the structure:
    Input → Conv1D(128) + BN + ReLU → Conv1D(256) + BN + ReLU → Conv1D(128) + BN + ReLU → Global pooling → Softmax
    """

    def __init__(self, input_size, num_classes):
        super(FCN, self).__init__()

        # Reshape input for 1D convolution (batch_size, 1, sequence_length)
        self.input_size = input_size

        # First convolutional block: 128 filters + BN + ReLU
        self.conv1 = nn.Conv2d(1, 128, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        # Second convolutional block: 256 filters + BN + ReLU
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        # Third convolutional block: 128 filters + BN + ReLU
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Final classification layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Reshape input to (batch_size, channels, length)
        x = x.unsqueeze(1)

        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, channels)

        # Classification
        x = self.fc(x)

        return x
