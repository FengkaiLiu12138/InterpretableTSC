import numpy as np
import pandas as pd


def _generate_labels(data, label_type, window_size=600):
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size

    # Initialize labels
    if label_type == 0:
        # Handle uni-variate TS
        n = len(data)
        half_window = window_size // 2
        labels = np.zeros(n, dtype=int)

        for i in range(half_window, n - half_window):
            window = data[i - half_window:i + half_window + 1]
            center_value = window[half_window]

            # Check if center is maximum or minimum in window
            if center_value == np.max(window):
                labels[i] = 1
            elif center_value == np.min(window):
                labels[i] = 1
        return labels

    elif label_type == 1:
        # Handle multi-variate TS
        if isinstance(data, pd.DataFrame):
            data = data.values

        n_samples, n_features = data.shape
        half_window = window_size // 2
        labels = np.zeros(n_samples, dtype=int)

        for i in range(half_window, n_samples - half_window):
            is_anomaly = False

            # Check each dimension/feature
            for feat_idx in range(n_features):
                feature_data = data[:, feat_idx]
                window = feature_data[i - half_window:i + half_window + 1]
                center_value = window[half_window]

                # If center point is max or min in any dimension, mark as anomaly
                if (center_value == np.max(window)) or (center_value == np.min(window)):
                    is_anomaly = True
                    break

            if is_anomaly:
                labels[i] = 1

        return labels
    else:
        raise ValueError(f"Unsupported label_type: {label_type}")


class DatasetConverter:
    def __init__(self, file_path, save_path=None):
        self.file_path = file_path
        self.labels = None
        self.data = None
        self.save_path = save_path
        self.column_names = ['Close', 'High', 'Low', 'Open', 'Volume', 'Labels']

    def convert(self, label_type=0, window_size=600, normalize=True, volume=True):
        """
        Load the dataset from the specified file path.
        If the file contains label columns, they will be used as labels directly.

        Label_type:
        0: Use Close price as label
        1: Use all data as labels

        window_size: Size of the window for generating labels

        normalize: Whether to normalize the data by columns into [0, 1]

        volume: Whether to include volume in the dataset
        """
        try:
            # Load the dataset. Some CSVs in this project do not have a header
            # row. We first attempt the default pandas loading behaviour. If the
            # inferred column names look like data entries (e.g. a date string),
            # we reload with ``header=None`` and assign proper column names.
            data = pd.read_csv(self.file_path)
            first_col = str(data.columns[0])
            if first_col[0].isdigit() and "-" in first_col:
                data = pd.read_csv(self.file_path, header=None)
                if data.shape[1] >= 6:
                    # Assume the first column is date/time and drop it
                    data.columns = [
                        "Date",
                        "Close",
                        "High",
                        "Low",
                        "Open",
                        "Volume",
                    ][: data.shape[1]]
                else:
                    data.columns = list(range(data.shape[1]))

            # Check if the dataset has label columns
            if 'Labels' in data.columns:
                print("Loading from existing labelled dataset...")
                # Already has labels
                self.data = data[self.column_names[:-1]].copy()
                self.labels = data['Labels'].values
            else:
                print("Loading and converting dataset...")
                # No labels, need to generate them
                if len(data.columns) >= 5:
                    # Assuming the data has at least 5 columns for financial data
                    self.data = data[data.columns[1:]].copy()
                    if volume:
                        self.data.columns = self.column_names[:-1]  # Rename columns
                    else:
                        self.data = self.data.iloc[:, :-1]
                    numerical_data = self.data.values
                    self.labels = _generate_labels(numerical_data, label_type, window_size)
                    print(data.head())
                else:
                    raise ValueError(f"Expected at least 5 columns for financial data, but got {len(data.columns)}")

            # Normalize the data by columns into [0, 1]
            if normalize:
                for col in self.data.columns:
                    min_val = self.data[col].min()
                    max_val = self.data[col].max()
                    if max_val > min_val:  # Avoid division by zero
                        self.data[col] = (self.data[col] - min_val) / (max_val - min_val)
                    else:
                        self.data[col] = 0  # Set to constant if min equals max

            # Add labels to the dataframe
            self.data['Labels'] = self.labels

            print("Data sample:")
            print(self.data.head())
            print(
                f"Data loaded successfully. Number of samples: {self.data.shape[0]}, Number of features: {self.data.shape[1] - 1}")
            print(f"Number of Positive samples: {np.sum(self.data['Labels'] == 1)}")

            # Save the converted dataset to a new CSV file
            if self.save_path:
                self.data.to_csv(self.save_path, index=False)
                print(f"Converted dataset saved to {self.save_path}")

            return self.data
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    dc = DatasetConverter(
        file_path="../Dataset/ftse_minute_data_may.csv",
        save_path="../Dataset/ftse_minute_data_may_labelled.csv"
    )

    dc.convert(label_type=1, volume=False)