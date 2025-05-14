import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multivariate MLP model:
      Input shape: (B, T, C)
        1) Flatten first: (B, T*C)
        2) Dropout(0.1) → Linear(500) + ReLU → Dropout(0.2)
        3) Linear(500) + ReLU → Dropout(0.2)
        4) Linear(500) + ReLU → Dropout(0.3)
        5) Linear(..., num_classes)

    Notes:
      - T = window_size
      - C = n_vars
      - The overall idea is the same as your original univariate MLP,
        except that the multivariate time series is automatically flattened before being fed in.
    """

    def __init__(self, window_size: int, n_vars: int, num_classes: int):
        """
        Args:
            window_size (int): Sequence length (window size), i.e. T
            n_vars (int): Feature dimension at each time step, i.e. C
            num_classes (int): Number of classes
        """
        super().__init__()

        input_size = window_size * n_vars  # Dimension after flattening

        self.model = nn.Sequential(
            nn.Dropout(0.1),                    # Input layer Dropout

            nn.Linear(input_size, 500),         # First hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(500, 500),                # Second hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(500, 500),                # Third hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(500, num_classes)         # Output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (B, T, C)
          → Flatten: (B, T*C)
          → Pass through several fully connected layers
        """
        # (B, T*C)
        x = x.view(x.size(0), -1)

        # Forward through MLP
        return self.model(x)
