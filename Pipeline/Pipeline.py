import copy
from BaselineModel.MLP_baseline import MLP
from Tools.DatasetConverter import DatasetConverter
from Tools.FTSE_dataset import FTSEDataCatcher
import datetime as dt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
import os


class Pipeline:
    def __init__(self, model, file_path, save_path=None):
        self.model = model
        self.file_path = file_path
        self.save_path = save_path
        self.dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.best_model = None
        self.window_size = None
        self.confusion_mat = None
        self.y_pred = None
        self.y_true = None
        self.X_test = None

        # Safely detect model dimensions
        self.input_size = None
        self.num_classes = None

        # Find input and output dimensions by checking the model layers
        if hasattr(model, 'model'):
            # Find first Linear layer for input size
            for layer in model.model:
                if isinstance(layer, nn.Linear):
                    self.input_size = layer.in_features
                    break

            # Find last Linear layer for output size
            for layer in reversed(list(model.model)):
                if isinstance(layer, nn.Linear):
                    self.num_classes = layer.out_features
                    break

        print(f"Detected model dimensions: input_size={self.input_size}, num_classes={self.num_classes}")

    def preprocessing(self, label_type, window_size, normalize):
        """
        Preprocess the dataset for training and evaluation.
        :param label_type: 0 for uni-variate, 1 for multi-variate
        :param window_size: defines the size of the sliding window
        :param normalize: normalize the data or not
        """
        # Convert the dataset to the required format
        dataset_converter = DatasetConverter(self.file_path, self.save_path)
        self.dataset = dataset_converter.convert(label_type, window_size, normalize)
        print(f"Preprocessed dataset shape: {self.dataset.shape}")
        self.window_size = window_size

    def data_loader(self, batch_size=32, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
        """
        Split dataset into train, validation and test sets and create DataLoader objects.

        :param batch_size: batch size for DataLoader
        :param train_ratio: ratio of training data
        :param valid_ratio: ratio of validation data
        :param test_ratio: ratio of test data
        """
        if self.dataset is None:
            raise ValueError("Dataset not preprocessed. Call preprocessing() first.")

        # Read the dataset
        if self.save_path and os.path.exists(self.save_path):
            df = pd.read_csv(self.save_path)
        else:
            df = self.dataset

        print(f"Dataset shape: {df.shape}")

        # Extract features and labels
        X = df.drop('Labels', axis=1).values  # Features
        y = df['Labels'].values  # Labels

        # Reshape X for MLP input if needed (time series to 2D)
        if len(X.shape) > 2:  # If X is 3D (samples, time_steps, features)
            X = X.reshape(X.shape[0], -1)  # Flatten to 2D (samples, time_steps*features)

        # Get the input size for the MLP model (number of features)
        self.input_size = X.shape[1]
        print(f"Input shape for MLP: {X.shape}, features dimension: {self.input_size}")

        # Update model if needed
        if hasattr(self.model, 'model'):
            # Find the first Linear layer and check its input dimension
            first_linear = None
            for layer in self.model.model:
                if isinstance(layer, nn.Linear):
                    first_linear = layer
                    break

            if first_linear and first_linear.in_features != self.input_size:
                print(f"Adjusting model input size from {first_linear.in_features} to {self.input_size}")
                # Create a new model with the correct input size
                self.model = MLP(input_size=self.input_size, num_classes=self.num_classes).to(self.device)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Create TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)

        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        valid_size = int(valid_ratio * total_size)
        test_size = total_size - train_size - valid_size

        # Split the dataset
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, [train_size, valid_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)

        print(
            f"Data split complete: {train_size} training samples, {valid_size} validation samples, {test_size} test samples")

        # Save test data for visualization
        test_data = []
        test_labels = []
        for inputs, labels in test_dataset:
            test_data.append(inputs)
            test_labels.append(labels)

        self.X_test = torch.stack(test_data)
        self.y_true = torch.stack(test_labels)

        return self.train_loader, self.valid_loader, self.test_loader

    def train(self, epochs=50, batch_size=32, learning_rate=0.001, weight_decay=0.0001,
              use_hyperparameter_tuning=False, early_stopping_patience=10):
        """
        Train the model on the preprocessed dataset with optional hyperparameter tuning.

        :param epochs: number of training epochs
        :param batch_size: size of each training batch
        :param learning_rate: learning rate for optimizer
        :param weight_decay: L2 regularization term
        :param use_hyperparameter_tuning: whether to use grid search for hyperparameter tuning
        :param early_stopping_patience: number of epochs to wait before early stopping
        :return: trained model and training history
        """
        if self.train_loader is None or self.valid_loader is None:
            # Create data loaders if not already done
            self.data_loader(batch_size=batch_size)

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        if use_hyperparameter_tuning:
            print("Starting hyperparameter tuning...")
            # Define hyperparameter grid
            param_grid = {
                'learning_rate': [0.01, 0.001, 0.0001],
                'weight_decay': [0.1, 0.01, 0.001, 0.0001],
                'optimizer': ['adam', 'sgd']
            }

            best_val_loss = float('inf')
            best_params = {}

            # Iterate over parameter combinations
            for params in ParameterGrid(param_grid):
                print(f"Trying parameters: {params}")

                # Reset model weights
                self.model.apply(self._reset_weights)

                # Configure optimizer based on params
                if params['optimizer'] == 'adam':
                    optimizer = optim.Adam(self.model.parameters(),
                                           lr=params['learning_rate'],
                                           weight_decay=params['weight_decay'])
                else:
                    optimizer = optim.SGD(self.model.parameters(),
                                          lr=params['learning_rate'],
                                          weight_decay=params['weight_decay'])

                # Train for a few epochs to evaluate this parameter set
                eval_epochs = 10  # Reduced number of epochs for hyperparameter search
                val_loss = self._train_loop(optimizer, criterion, eval_epochs, early_stopping_patience)

                # Update best parameters if this combination is better
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = params

            print(f"Best parameters found: {best_params}")

            # Reset model and train with best parameters
            self.model.apply(self._reset_weights)

            # Configure optimizer with best parameters
            if best_params['optimizer'] == 'adam':
                optimizer = optim.Adam(self.model.parameters(),
                                       lr=best_params['learning_rate'],
                                       weight_decay=best_params['weight_decay'])
            else:
                optimizer = optim.SGD(self.model.parameters(),
                                      lr=best_params['learning_rate'],
                                      weight_decay=best_params['weight_decay'])

        else:
            # Use the provided hyperparameters
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=learning_rate,
                                   weight_decay=weight_decay)

        # Train the model with the selected hyperparameters
        val_loss = self._train_loop(optimizer, criterion, epochs, early_stopping_patience)

        return self.best_model, val_loss

    def _reset_weights(self, m):
        """Reset model weights for hyperparameter tuning"""
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def _train_loop(self, optimizer, criterion, epochs, early_stopping_patience):
        """Internal training loop used by the train method"""
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}

        self.model.to(self.device)

        # Save a copy of the initial model
        if self.input_size is not None and self.num_classes is not None:
            self.best_model = MLP(
                input_size=self.input_size,
                num_classes=self.num_classes
            )
            self.best_model.load_state_dict(self.model.state_dict())
            self.best_model.to(self.device)
        else:
            print("Warning: Could not determine model input/output sizes. Using direct model copy.")
            self.best_model = copy.deepcopy(self.model)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss = train_loss / len(self.train_loader.dataset)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

            val_loss = val_loss / len(self.valid_loader.dataset)

            # Save history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)

            # Print training progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the best model
                self.best_model.load_state_dict(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break

        # Restore the best model
        self.model.load_state_dict(self.best_model.state_dict())

        return best_val_loss

    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        :return: dictionary containing evaluation metrics (accuracy, F1 score, confusion matrix)
        """
        if self.test_loader is None:
            raise ValueError("Test data not loaded. Call data_loader() first.")

        self.model.eval()
        self.model.to(self.device)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                # Store predictions and true labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert lists to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Store for visualization
        self.y_pred = all_preds

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        self.confusion_mat = confusion_matrix(all_labels, all_preds)

        # Print metrics
        print(f"\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Print confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.confusion_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': self.confusion_mat
        }

    def visualization(self):
        """
        Visualize examples of TP, FP, TN, FN cases from the test set.
        Creates line plots for time series with arrow marking the prediction point.
        """
        if self.y_pred is None or self.X_test is None or self.y_true is None:
            raise ValueError("Evaluation not performed. Call evaluate() first.")

        # Convert tensors to numpy if needed
        if isinstance(self.X_test, torch.Tensor):
            X_test_np = self.X_test.cpu().numpy()
        else:
            X_test_np = self.X_test

        if isinstance(self.y_true, torch.Tensor):
            y_true_np = self.y_true.cpu().numpy()
        else:
            y_true_np = self.y_true

        # Find indices for TP, FP, TN, FN
        tp_indices = np.where((y_true_np == 1) & (self.y_pred == 1))[0]
        fp_indices = np.where((y_true_np == 0) & (self.y_pred == 1))[0]
        tn_indices = np.where((y_true_np == 0) & (self.y_pred == 0))[0]
        fn_indices = np.where((y_true_np == 1) & (self.y_pred == 0))[0]

        # Check if we have examples for each category
        categories_available = {
            'TP': len(tp_indices) > 0,
            'FP': len(fp_indices) > 0,
            'TN': len(tn_indices) > 0,
            'FN': len(fn_indices) > 0
        }

        missing_categories = [cat for cat, available in categories_available.items() if not available]
        if missing_categories:
            print(f"Warning: Not enough examples for categories: {', '.join(missing_categories)}")

        # Define feature names
        feature_names = ['Close', 'High', 'Low', 'Open', 'Volume']

        # Get original time series data - this requires access to the original dataset
        # We'll need to import pandas and load the data
        try:
            # Load the original dataset to get time context
            if self.save_path and os.path.exists(self.save_path):
                full_data = pd.read_csv(self.save_path)
            else:
                print("Warning: Original dataset not found. Using test samples directly.")
                full_data = None

            # Create a figure for visualization - 2x2 grid for TP, FP, TN, FN
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle('Time Series Visualization by Category', fontsize=18)

            # Define categories to plot
            categories = [
                ('TP (True Positive)', tp_indices, 0, 0),
                ('FP (False Positive)', fp_indices, 0, 1),
                ('TN (True Negative)', tn_indices, 1, 0),
                ('FN (False Negative)', fn_indices, 1, 1)
            ]

            # Function to get time series context for a test example
            def get_time_series_context(idx, context_size=30):
                """
                Get time series context around the test example.
                For simplicity, creates synthetic time series if original data not available.

                Args:
                    idx: Index of the test example
                    context_size: Number of time steps to include in context (half before, half after)

                Returns:
                    time_series_data: Dictionary of feature time series
                """
                # If we can locate the example in the original dataset
                if full_data is not None:
                    # Here we would need to map the test index back to the original dataset
                    # This is a simplification - in practice you need a proper mapping
                    ts_data = {}

                    # Create a time window around the point
                    # In a real implementation, you'd extract the proper time window from the dataset
                    # This is just a placeholder approach
                    for i, feature in enumerate(feature_names):
                        # Extract sample value for this feature
                        sample_value = X_test_np[idx][i]

                        # Generate synthetic time series around this point
                        # In a real implementation, you'd use the actual time series data
                        ts = np.linspace(sample_value * 0.8, sample_value * 1.2, context_size)
                        # Make the center point the actual value
                        mid_point = context_size // 2
                        ts[mid_point] = sample_value
                        ts_data[feature] = ts

                    return ts_data
                else:
                    # Create synthetic time series if original data not available
                    ts_data = {}
                    for i, feature in enumerate(feature_names):
                        if i < len(X_test_np[idx]):
                            # Extract sample value for this feature
                            sample_value = X_test_np[idx][i]

                            # Generate synthetic time series around this point
                            ts = np.random.normal(sample_value, sample_value * 0.1, context_size)
                            # Make the center point the actual value
                            mid_point = context_size // 2
                            ts[mid_point] = sample_value
                            ts_data[feature] = ts

                    return ts_data

            # Plot each category
            for category, indices, row, col in categories:
                ax = axes[row, col]
                ax.set_title(category, fontsize=14)

                if len(indices) > 0:
                    # Get the first example for this category
                    idx = indices[0]

                    # Get time series context
                    time_context = 30  # Number of time steps
                    ts_data = get_time_series_context(idx, time_context)

                    # Time steps array
                    time_steps = np.arange(time_context)
                    middle_point = time_context // 2

                    # Plot each feature as a separate line
                    for feature, values in ts_data.items():
                        line = ax.plot(time_steps, values, label=feature, linewidth=2)

                        # Mark the prediction point with an arrow
                        ax.annotate('',
                                    xy=(middle_point, values[middle_point]),
                                    xytext=(middle_point, values[middle_point] * 1.15),
                                    arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                                    horizontalalignment='center')

                    # Add vertical line at prediction point
                    ax.axvline(x=middle_point, color='gray', linestyle='--', alpha=0.7)

                    # Configure the plot
                    ax.set_xlabel('Time Step', fontsize=12)
                    ax.set_ylabel('Value', fontsize=12)
                    ax.legend(loc='upper right')
                    ax.grid(True, linestyle='--', alpha=0.7)

                    # Add annotation for true and predicted labels
                    label_text = f"True: {y_true_np[idx]}, Pred: {self.y_pred[idx]}"
                    ax.text(0.05, 0.95, label_text, transform=ax.transAxes,
                            fontsize=12, fontweight='bold', verticalalignment='top')
                else:
                    ax.text(0.5, 0.5, f'No {category} examples available',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=14)

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()

            # Create a table with examples
            print("\nExample cases table:")
            print("-" * 80)
            print(
                f"{'Category':<15} | {'True Label':<10} | {'Predicted':<10} | {'Features (Close, High, Low, Open, Volume)'}")
            print("-" * 80)

            category_data = [
                ("TP", tp_indices),
                ("FP", fp_indices),
                ("TN", tn_indices),
                ("FN", fn_indices)
            ]

            for category, indices in category_data:
                if len(indices) > 0:
                    idx = indices[0]
                    true_label = y_true_np[idx]
                    pred_label = self.y_pred[idx]

                    # Format feature values
                    feature_values = X_test_np[idx]
                    if len(feature_values) >= 5:
                        feature_str = ", ".join([f"{feature_values[i]:.4f}" for i in range(5)])
                    else:
                        feature_str = "Data shape doesn't match expected features"

                    print(f"{category:<15} | {true_label:<10} | {pred_label:<10} | {feature_str}")
                else:
                    print(f"{category:<15} | {'N/A':<10} | {'N/A':<10} | {'No examples available'}")

            print("-" * 80)

            # Return information about examples found
            return {
                'tp_example': tp_indices[0] if len(tp_indices) > 0 else None,
                'fp_example': fp_indices[0] if len(fp_indices) > 0 else None,
                'tn_example': tn_indices[0] if len(tn_indices) > 0 else None,
                'fn_example': fn_indices[0] if len(fn_indices) > 0 else None
            }

        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            import traceback
            traceback.print_exc()

            # Fall back to simple feature visualization
            plt.figure(figsize=(12, 8))
            plt.suptitle('Feature Values by Category', fontsize=16)

            # Plot available categories
            available_cats = []
            available_values = []

            for cat_name, indices in [("TP", tp_indices), ("FP", fp_indices),
                                      ("TN", tn_indices), ("FN", fn_indices)]:
                if len(indices) > 0:
                    available_cats.append(cat_name)
                    available_values.append(X_test_np[indices[0]])

            # If we have any available categories
            if available_cats:
                # Create a bar chart showing feature values for each category
                x = np.arange(len(feature_names))
                width = 0.8 / len(available_cats)

                for i, (cat, values) in enumerate(zip(available_cats, available_values)):
                    plt.bar(x + i * width - 0.4 + width / 2, values[:len(feature_names)],
                            width=width, label=cat)

                plt.xlabel('Features')
                plt.ylabel('Values')
                plt.xticks(x, feature_names)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()

            return {
                'tp_example': tp_indices[0] if len(tp_indices) > 0 else None,
                'fp_example': fp_indices[0] if len(fp_indices) > 0 else None,
                'tn_example': tn_indices[0] if len(tn_indices) > 0 else None,
                'fn_example': fn_indices[0] if len(fn_indices) > 0 else None
            }


if "__main__" == __name__:
    # Example usage
    model = MLP(input_size=5, num_classes=2)  # Adjust input size and number of classes as needed
    # file_path = "../Dataset/ftse_minute_data_daily.csv"
    # save_path = "../Dataset/ftse_minute_data_daily_labelled.csv"
    #
    # pipeline = Pipeline(model, file_path, save_path)
    pipeline = Pipeline(model, file_path="../Dataset/ftse_minute_data_daily_labelled.csv")
    pipeline.preprocessing(label_type=0, window_size=600, normalize=True)
    pipeline.data_loader(batch_size=32)
    pipeline.train(epochs=50, use_hyperparameter_tuning=False)
    pipeline.evaluate()
    pipeline.visualization()
