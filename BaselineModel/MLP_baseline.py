import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    多变量 MLP 模型:
      输入形状: (B, T, C)
        1) 先 flatten: (B, T*C)
        2) Dropout(0.1) → Linear(500) + ReLU → Dropout(0.2)
        3) Linear(500) + ReLU → Dropout(0.2)
        4) Linear(500) + ReLU → Dropout(0.3)
        5) Linear(..., num_classes)

    说明:
      - T = window_size
      - C = n_vars
      - 整体思路和你原有的单变量 MLP 相同，只是自动把多维时序 flatten 后输入。
    """

    def __init__(self, window_size: int, n_vars: int, num_classes: int):
        """
        Args:
            window_size (int): 序列长度 (滑窗长度), 即 T
            n_vars (int): 每个时间步的特征维度, 即 C
            num_classes (int): 分类类别数
        """
        super().__init__()

        input_size = window_size * n_vars  # flatten 后的维度

        self.model = nn.Sequential(
            nn.Dropout(0.1),                    # 输入层 Dropout

            nn.Linear(input_size, 500),         # First hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(500, 500),               # Second hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(500, 500),               # Third hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(500, num_classes)        # Output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, T, C)
          → flatten: (B, T*C)
          → 经过若干全连接层
        """
        # (B, T*C)
        x = x.view(x.size(0), -1)

        # forward through MLP
        return self.model(x)
