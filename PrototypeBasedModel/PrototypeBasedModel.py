from torch import nn

class PrototypeFeatureExtractor:
    def __init__(self, data, prototypes):
        """
        Args:
            data (torch.Tensor): 输入数据，形状为 (B, T, C)
            prototypes (torch.Tensor): 原型，形状为 (num_prototypes, C)
        """
        self.data = data
        self.prototypes = prototypes

    def compute_prototype_features(self, metric='euclidean'):
        """
        计算原型特征
        Args:
            metric (str): 距离度量方式，默认使用欧氏距离
            - 可选 'euclidean', 'dtw', 'cosine', 'mahalanobis'
        Returns:
            torch.Tensor: 原型特征，形状为 (B, num_prototypes)
        """
        # 计算原型特征
        if metric == 'euclidean':
            return self._compute_euclidean_features()
        else:
            raise ValueError(f"Unsupported metric: {metric}")

class PrototypeBasedModel(nn.Module):
    def __init__(self, window_size: int, n_vars: int, num_classes: int):
        """
        Args:
            window_size (int): 序列长度 (滑窗长度), 即 T
            n_vars (int): 每个时间步的特征维度, 即 C
            num_classes (int): 分类类别数
        """
        super().__init__()
