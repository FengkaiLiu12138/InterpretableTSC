import numpy as np


class PrototypeSelector:
    def __init__(self, data, labels, window_size=600):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.half_window = window_size // 2

    def select_prototypes(self, num_prototypes ,selection_type='random'):
        '''
        Select prototypes from the dataset based on the specified method and remove them from the dataset.
        :param num_prototypes: number of prototypes to select
        :param data: dataset
        :param selection_type:
        'random': Randomly select prototypes
        'k-means': K-means clustering to select prototypes
        'gmm': Gaussian mixture model
        :return: prototypes, remaining data
        '''
        if selection_type == 'random':
            return self.random_selection(num_prototypes)
        elif selection_type == 'k-means':
            return self.k_means_selection(num_prototypes)
        elif selection_type == 'gmm':
            return self.gmm_selection(num_prototypes)
        else:
            raise ValueError(f"Unsupported selection type: {selection_type}")

    def random_selection(self, num_prototypes):
        '''
        Randomly select prototypes from the dataset with roughly equal positive and negative samples.
        :param num_prototypes: number of prototypes to select
        :return: prototypes, remaining data
        '''
        # randomly select indices
        indices = np.random.choice(len(self.data), num_prototypes, replace=False)
        prototypes = self.data[indices]
        remaining_data = np.delete(self.data, indices, axis=0)



    def k_means_selection(self, num_prototypes):
        '''
        K-means clustering to select prototypes from the dataset.
        :param num_prototypes: number of prototypes to select
        :return: prototypes, remaining data
        '''

    def gmm_selection(self, num_prototypes):
        '''
        Gaussian mixture model to select prototypes from the dataset.
        :param num_prototypes: number of prototypes to select
        :return: prototypes, remaining data
        '''