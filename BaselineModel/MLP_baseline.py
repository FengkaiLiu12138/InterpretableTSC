import torch.nn as nn


class MLP(nn.Module):
    """
    MLP model as described in the paper:
    Input → Dropout(0.1) → Dense(500, ReLU) → Dropout(0.2) → Dense(500, ReLU) →
    Dropout(0.2) → Dense(500, ReLU) → Dropout(0.3) → Softmax output
    """

    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            # Input layer with dropout
            nn.Dropout(0.1),

            # First hidden layer: 500 neurons with ReLU
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Second hidden layer: 500 neurons with ReLU
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Third hidden layer: 500 neurons with ReLU
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(500, num_classes)
        )

    def forward(self, x):
        return self.model(x)
