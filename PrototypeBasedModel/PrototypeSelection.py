import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class PrototypeSelector:
    def __init__(self, data, labels, window_size=600):
        """
        data: shape (N, window_size, num_features) or (N, some_feature_dim)
        labels: shape (N,), binary (0 or 1)
        window_size: default 600
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.half_window = window_size // 2

    def select_prototypes(self, num_prototypes, selection_type='random'):
        """
        Select prototypes from the dataset based on the specified method and remove them from the dataset.

        Args:
            num_prototypes (int): number of prototypes to select
            selection_type (str): 'random' / 'k-means' / 'gmm'

        Returns:
            prototypes (ndarray): shape (num_prototypes, ...)
            prototype_labels (ndarray): shape (num_prototypes,)
            remaining_data (ndarray): shape (N - num_prototypes, ...)
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
        """
        Randomly select prototypes from the dataset, trying to get a roughly equal
        number of positive and negative samples (if possible).

        Returns:
            prototypes, prototype_labels, remaining_data, remaining_labels
        """
        # Separate indices for pos & neg
        pos_idx = np.where(self.labels == 1)[0]
        neg_idx = np.where(self.labels == 0)[0]

        # Shuffle
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        # Half from pos, half from neg (if possible)
        half = num_prototypes // 2
        num_pos = min(half, len(pos_idx))
        num_neg = num_prototypes - num_pos
        num_neg = min(num_neg, len(neg_idx))

        selected_pos = pos_idx[:num_pos]
        selected_neg = neg_idx[:num_neg]
        selected_idx = np.concatenate([selected_pos, selected_neg])

        # If we still haven't reached num_prototypes (e.g. minority class is too small)
        # We can randomly pick the remainder from whichever class has leftover.
        remainder = num_prototypes - len(selected_idx)
        if remainder > 0:
            # Combine leftover indices
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
        K-means clustering to select prototypes from the dataset. We cluster
        the entire dataset into 'num_prototypes' clusters. We then pick one sample
        (the closest to the cluster center) from each cluster as a prototype.

        Returns:
            prototypes, prototype_labels, remaining_data, remaining_labels
        """
        # Flatten each sample if needed. Example shape: (N, window_size*features)
        N = len(self.data)
        if self.data.ndim == 3:
            # (N, window_size, feature_dim) -> (N, window_size*feature_dim)
            flat_data = self.data.reshape(N, -1)
        else:
            # Assume data is already (N, some_feature_dim)
            flat_data = self.data

        kmeans = KMeans(n_clusters=num_prototypes, random_state=42)
        kmeans.fit(flat_data)
        centers = kmeans.cluster_centers_
        labels_km = kmeans.labels_  # cluster assignment for each sample

        prototypes = []
        prototype_labels = []
        chosen_indices = []

        for c_idx in range(num_prototypes):
            # samples in cluster c_idx
            cluster_indices = np.where(labels_km == c_idx)[0]
            if len(cluster_indices) == 0:
                # If a cluster is empty (rare but can happen), skip
                continue

            # find the sample closest to the center
            cluster_data = flat_data[cluster_indices]
            center = centers[c_idx]

            # dist^2 from center
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
        Gaussian mixture model to select prototypes from the dataset. Similar to K-means:
          - Fit GMM with 'num_prototypes' components
          - For each component, pick the sample closest to the component mean
            as the prototype.
        Returns:
            prototypes, prototype_labels, remaining_data, remaining_labels
        """
        # Flatten each sample if needed.
        N = len(self.data)
        if self.data.ndim == 3:
            flat_data = self.data.reshape(N, -1)
        else:
            flat_data = self.data

        gmm = GaussianMixture(n_components=num_prototypes, random_state=42)
        gmm.fit(flat_data)

        # Means of shape (num_prototypes, feature_dim)
        means = gmm.means_
        # Soft cluster assignment: not used here, but you could
        # predict which component each sample likely belongs to
        # comp_assign = gmm.predict(flat_data)

        prototypes = []
        prototype_labels = []
        chosen_indices = []

        for i in range(num_prototypes):
            mean_i = means[i]
            # find the sample closest to this component mean
            dists = np.sum((flat_data - mean_i) ** 2, axis=1)
            min_idx = np.argmin(dists)
            prototypes.append(self.data[min_idx])
            prototype_labels.append(self.labels[min_idx])
            chosen_indices.append(min_idx)

        chosen_indices = np.array(chosen_indices, dtype=int)
        prototypes = np.array(prototypes)
        prototype_labels = np.array(prototype_labels)

        # Build remaining
        mask = np.ones(len(self.data), dtype=bool)
        mask[chosen_indices] = False
        remaining_data = self.data[mask]
        remaining_labels = self.labels[mask]

        return prototypes, prototype_labels, remaining_data, remaining_labels
