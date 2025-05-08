import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    可灵活配置的 LSTM 模型，用于时序分类:
      输入形状: (B, T, C)
        - B: batch_size
        - T: 序列长度 (window_size)
        - C: 特征维度 (feature_dim)
      输出形状: (B, num_classes)

    默认:
      - 两层 LSTM (num_layers=2)
      - hidden_dim=128
      - dropout=0.2
      - 非双向 (bidirectional=False)

    你可根据需要调整这些参数，以寻求更好的效果。
    """

    def __init__(
            self,
            window_size: int,  # 序列长度, 也可以不用实际存下来
            n_vars: int,  # 每个时间步的特征数
            num_classes: int,  # 分类数
            hidden_dim: int = 128,  # 隐藏层维度
            num_layers: int = 2,  # LSTM 堆叠层数
            dropout: float = 0.2,  # LSTM 内部的 dropout
            bidirectional: bool = False  # 是否双向
    ):
        super().__init__()
        self.window_size = window_size
        self.n_vars = n_vars
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # batch_first=True -> 输入/输出形状均为 (B, T, input_size)
        # 注意: 如果是双向，需要在后续的全连接中考虑隐藏维度×2
        self.lstm = nn.LSTM(
            input_size=n_vars,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # 当 LSTM 层数>1 时，才会应用 dropout
            bidirectional=bidirectional
        )

        # 如果是双向 LSTM，最终的输出维度是 hidden_dim * 2
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # 全连接层，将 LSTM 的最后一步输出映射到 num_classes
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x 形状: (B, T, C)
          - B: batch_size
          - T: 序列长度
          - C: 特征维度

        Returns: logits (B, num_classes)
        """
        # 走 LSTM 前向传播
        # out: (B, T, hidden_dim * num_directions)
        # (h, c): (num_layers * num_directions, B, hidden_dim)
        out, (h, c) = self.lstm(x)

        # 我们这里用最后时刻 T 的输出做分类
        # out[:, -1, :] => (B, hidden_dim * num_directions)
        last_output = out[:, -1, :]

        # 线性映射到 num_classes
        logits = self.fc(last_output)  # (B, num_classes)
        return logits
