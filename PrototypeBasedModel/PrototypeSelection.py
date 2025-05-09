import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

###############################################################################
#  (2) 原型选择器: PrototypeSelector
###############################################################################
class PrototypeSelector:
    """
    根据指定方法(随机 / K-means / GMM)从数据集中选出指定数量的原型，
    并移除这些原型样本，返回 (prototypes, prototype_labels, remaining_data, remaining_labels)。
    这里的 data 形状通常是 (N, 600, n_features)，
    因为我们每条样本是一段 time series (T=600) × n_vars (e.g. 5)。
    """
    def __init__(self, data, labels, window_size=600):
        """
        data: shape (N, window_size, num_features) -> (N, 600, 5)
        labels: shape (N,)
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.half_window = window_size // 2

    def select_prototypes(self, num_prototypes, selection_type='random'):
        """
        Select prototypes from the dataset by 'random' / 'k-means' / 'gmm'.

        Returns:
            prototypes (ndarray): shape (num_prototypes, window_size, n_features)
            prototype_labels (ndarray): shape (num_prototypes,)
            remaining_data (ndarray): shape (N - num_prototypes, window_size, n_features)
            remaining_labels (ndarray): shape (N - num_prototypes,)
        """
        if selection_type == 'random':
            return self.random_selection(num_prototypes)
        elif selection_type == 'k-means':
            return self.k_means_selection(num_prototypes)
        elif selection_type == 'gmm':
            return self.gmm_selection(num_prototypes)
        else:
            raise ValueError(f"Unsupported selection type: {selection_type}")

    def random_selection(self, num_prototypes):
        """随机选出指定数量的原型，尝试保证正负样本平衡。"""
        # Separate indices for pos & neg
        pos_idx = np.where(self.labels == 1)[0]
        neg_idx = np.where(self.labels == 0)[0]

        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        half = num_prototypes // 2
        num_pos = min(half, len(pos_idx))
        num_neg = num_prototypes - num_pos
        num_neg = min(num_neg, len(neg_idx))

        selected_pos = pos_idx[:num_pos]
        selected_neg = neg_idx[:num_neg]
        selected_idx = np.concatenate([selected_pos, selected_neg])

        # 如果还不够 num_prototypes，就继续从剩余样本中随机补
        remainder = num_prototypes - len(selected_idx)
        if remainder > 0:
            leftover_idx = np.concatenate([pos_idx[num_pos:], neg_idx[num_neg:]])
            np.random.shuffle(leftover_idx)
            selected_idx = np.concatenate([selected_idx, leftover_idx[:remainder]])

        # Extract prototypes
        prototypes = self.data[selected_idx]
        prototype_labels = self.labels[selected_idx]

        # Build remaining
        mask = np.ones(len(self.data), dtype=bool)
        mask[selected_idx] = False
        remaining_data = self.data[mask]
        remaining_labels = self.labels[mask]

        return prototypes, prototype_labels, remaining_data, remaining_labels

    def k_means_selection(self, num_prototypes):
        """
        K-means: 先把所有样本打平成 (N, 600*n_features)，聚成 num_prototypes 类；
        再从每个聚类里选离聚类中心最近的样本作为原型。
        """
        N = len(self.data)
        # flatten => shape (N, 600*n_features)
        flat_data = self.data.reshape(N, -1)

        kmeans = KMeans(n_clusters=num_prototypes, random_state=42)
        kmeans.fit(flat_data)
        centers = kmeans.cluster_centers_
        labels_km = kmeans.labels_  # cluster assignment for each sample

        prototypes = []
        prototype_labels = []
        chosen_indices = []

        for c_idx in range(num_prototypes):
            cluster_indices = np.where(labels_km == c_idx)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_data = flat_data[cluster_indices]
            center = centers[c_idx]
            dists = np.sum((cluster_data - center) ** 2, axis=1)
            min_idx = np.argmin(dists)
            best_sample_global_idx = cluster_indices[min_idx]

            prototypes.append(self.data[best_sample_global_idx])
            prototype_labels.append(self.labels[best_sample_global_idx])
            chosen_indices.append(best_sample_global_idx)

        chosen_indices = np.array(chosen_indices, dtype=int)
        prototypes = np.array(prototypes)
        prototype_labels = np.array(prototype_labels)

        # Build remaining
        mask = np.ones(len(self.data), dtype=bool)
        mask[chosen_indices] = False
        remaining_data = self.data[mask]
        remaining_labels = self.labels[mask]

        return prototypes, prototype_labels, remaining_data, remaining_labels

    def gmm_selection(self, num_prototypes):
        """
        GMM: 跟 k-means 类似，只是用高斯混合模型聚类。
        """
        N = len(self.data)
        flat_data = self.data.reshape(N, -1)

        gmm = GaussianMixture(n_components=num_prototypes, random_state=42)
        gmm.fit(flat_data)

        means = gmm.means_
        # 如果需要查看每个样本的组件归属，可用 gmm.predict(flat_data)

        prototypes = []
        prototype_labels = []
        chosen_indices = []

        for i in range(num_prototypes):
            mean_i = means[i]
            dists = np.sum((flat_data - mean_i) ** 2, axis=1)
            min_idx = np.argmin(dists)
            prototypes.append(self.data[min_idx])
            prototype_labels.append(self.labels[min_idx])
            chosen_indices.append(min_idx)

        chosen_indices = np.array(chosen_indices, dtype=int)
        prototypes = np.array(prototypes)
        prototype_labels = np.array(prototype_labels)

        mask = np.ones(len(self.data), dtype=bool)
        mask[chosen_indices] = False
        remaining_data = self.data[mask]
        remaining_labels = self.labels[mask]

        return prototypes, prototype_labels, remaining_data, remaining_labels

