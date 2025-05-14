import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Configurable LSTM model for sequence classification.
    Input shape: (B, T, C)
        - B: batch size
        - T: sequence length (window_size)
        - C: feature dimension (n_vars)
    Output shape: (B, num_classes)

    Default settings:
        - 2 LSTM layers (num_layers = 2)
        - hidden_dim = 128
        - dropout = 0.2
        - Unidirectional (bidirectional = False)

    You can adjust these parameters to achieve better performance.
    """

    def __init__(
            self,
            window_size: int,      # Sequence length (can be stored or ignored)
            n_vars: int,           # Number of features at each time step
            num_classes: int,      # Number of output classes
            hidden_dim: int = 128, # Hidden dimension size
            num_layers: int = 2,   # Number of stacked LSTM layers
            dropout: float = 0.2,  # Dropout inside the LSTM
            bidirectional: bool = False  # Use bidirectional LSTM if True
    ):
        super().__init__()
        self.window_size = window_size
        self.n_vars = n_vars
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # batch_first=True -> input/output shape is (B, T, input_size)
        # Note: if bidirectional, the hidden dimension is doubled in the fully connected layer
        self.lstm = nn.LSTM(
            input_size=n_vars,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout is applied only when num_layers > 1
            bidirectional=bidirectional
        )

        # Output dimension of the LSTM is hidden_dim × 2 if bidirectional
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Fully connected layer maps the last time step to num_classes
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, C)
        Returns:
            Tensor: Logits of shape (B, num_classes)
        """
        # Forward pass through the LSTM
        # out: (B, T, hidden_dim * num_directions)
        out, (h, c) = self.lstm(x)

        # Use the output at the last time step for classification
        # out[:, -1, :] -> (B, hidden_dim * num_directions)
        last_output = out[:, -1, :]

        # Linear projection to num_classes
        logits = self.fc(last_output)  # (B, num_classes)
        return logits
