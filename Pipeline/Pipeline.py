import copy
import os
from typing import Literal

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils import resample

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from BaselineModel import (
    ResNet_baseline,
    CNN_baseline,
    LSTM_baseline,
    MLP_baseline,
    FCN_baseline
)

from PrototypeBasedModel import PrototypeBasedModel
from PrototypeBasedModel.PrototypeBasedModel import PrototypeBasedModel, PrototypeFeatureExtractor, PrototypeSelector


###############################################################################
#  Pipeline
###############################################################################
class Pipeline:
    """
    An end-to-end pipeline for binary classification (0/1) with optional prototype features.
    Only the training set is balanced; validation and test sets remain untouched.
    """

    def __init__(
        self,
        model_class,
        file_path: str,
        n_vars: int,
        num_classes: int,
        result_dir: str | None = None,
        use_prototype: bool = False,
        num_prototypes: int = 8,
        prototype_selection_type: str = 'random',
        prototype_distance_metric: str = 'euclidean',
    ):
        self.model_class = model_class
        self.file_path = file_path
        self.n_vars = n_vars
        self.num_classes = num_classes
        self.window_size = 600
        self.use_prototype = use_prototype
        self.num_prototypes = num_prototypes
        self.prototype_selection_type = prototype_selection_type
        self.prototype_distance_metric = prototype_distance_metric

        if result_dir is None:
            result_dir = f"../Result/{model_class.__name__}"
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} | result_dir={self.result_dir}")

        self._df = None
        self.dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.model = None
        self.best_model = None
        self._opt_metric = "loss"

        self._normalize = True
        self._balance = False
        self._balance_strategy = "over"

        self.X_test = None
        self.y_true = None
        self.confusion_mat = None

        self._prototypes = None
        self._proto_labels = None

        self._load_raw_csv()

    def _load_raw_csv(self):
        df = pd.read_csv(self.file_path)
        self._df = df.copy()

    @staticmethod
    def _reset_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            m.reset_parameters()

    @staticmethod
    def _binary_f1(labels: np.ndarray, preds: np.ndarray) -> float:
        return f1_score(labels, preds, pos_label=1, average="binary")

    def preprocessing(self, normalize: bool = True):
        """
        Basic preprocessing (e.g. normalization and optional prototype extraction).
        """
        self._normalize = normalize
        df = self._df.copy()
        feature_cols = ["Close", "High", "Low", "Open", "Volume"][:self.n_vars]
        if normalize:
            for col in feature_cols:
                mn, mx = df[col].min(), df[col].max()
                df[col] = (df[col] - mn) / (mx - mn + 1e-12)

        X_list, y_list = [], []
        half_w = self.window_size // 2
        total_len = len(df)
        for start_idx in range(total_len - self.window_size + 1):
            end_idx = start_idx + self.window_size
            window_data = df.iloc[start_idx:end_idx][feature_cols].values
            center_label_idx = start_idx + half_w
            label = df.iloc[center_label_idx]["Labels"]
            X_list.append(window_data)
            y_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        if self.use_prototype:
            selector = PrototypeSelector(X, y, window_size=self.window_size)
            protos, proto_labels, rem_data, rem_labels = selector.select_prototypes(
                num_prototypes=self.num_prototypes,
                selection_type=self.prototype_selection_type
            )
            self._prototypes = protos
            self._proto_labels = proto_labels
            t_data = torch.from_numpy(rem_data)
            t_proto = torch.from_numpy(protos)
            extractor = PrototypeFeatureExtractor(t_data, t_proto)
            extractor.plot_prototype_feature_map(
                metric=self.prototype_distance_metric,
                save_path=os.path.join(self.result_dir, "prototype_feature_map.png")
            )
            feats = extractor.compute_prototype_features(metric=self.prototype_distance_metric)
            X = feats.numpy()
            y = rem_labels

        self.dataset = (X, y)

    def data_loader(
        self,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        balance: bool = False,
        balance_strategy: Literal["over", "under"] = "over",
    ):
        """
        Split into train/valid/test, then optionally balance only the training set.
        """
        self._balance = balance
        self._balance_strategy = balance_strategy

        if self.dataset is None:
            raise ValueError("Call preprocessing() first.")

        X, y = self.dataset
        print(f"Full dataset shape: X={X.shape}, y={y.shape}")
        full_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

        total = len(full_dataset)
        tr = int(train_ratio * total)
        va = int(valid_ratio * total)
        te = total - tr - va
        train_ds, valid_ds, test_ds = random_split(
            full_dataset, [tr, va, te], generator=torch.Generator().manual_seed(42)
        )

        print(f"Split sizes: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")

        if balance:
            x_train_list, y_train_list = [], []
            for (xx, yy) in train_ds:
                x_train_list.append(xx.numpy())
                y_train_list.append(yy.item())
            x_train_arr = np.array(x_train_list)
            y_train_arr = np.array(y_train_list)

            c0 = np.sum(y_train_arr == 0)
            c1 = np.sum(y_train_arr == 1)
            if c1 > c0:
                maj_label, min_label = 1, 0
            else:
                maj_label, min_label = 0, 1

            x_maj = x_train_arr[y_train_arr == maj_label]
            x_min = x_train_arr[y_train_arr == min_label]
            if balance_strategy == "over":
                x_min_rs = resample(x_min, replace=True, n_samples=len(x_maj), random_state=42)
                y_min_rs = np.full(len(x_maj), min_label)
                x_bal = np.concatenate([x_maj, x_min_rs], axis=0)
                y_bal = np.concatenate([
                    np.full(len(x_maj), maj_label),
                    y_min_rs
                ], axis=0)
            else:
                x_maj_rs = resample(x_maj, replace=False, n_samples=len(x_min), random_state=42)
                y_maj_rs = np.full(len(x_min), maj_label)
                x_bal = np.concatenate([x_maj_rs, x_min], axis=0)
                y_bal = np.concatenate([y_maj_rs, np.full(len(x_min), min_label)], axis=0)

            shuffle_idx = np.random.RandomState(42).permutation(len(x_bal))
            x_bal = x_bal[shuffle_idx]
            y_bal = y_bal[shuffle_idx]
            print(f"Balanced train set: X={x_bal.shape}, y={y_bal.shape}")
            train_ds = TensorDataset(torch.from_numpy(x_bal).float(), torch.from_numpy(y_bal).long())
        else:
            # Print train set shape if not balanced
            x_train_list, y_train_list = [], []
            for (xx, yy) in train_ds:
                x_train_list.append(xx.numpy())
                y_train_list.append(yy.item())
            print(f"Unbalanced train set: X={np.array(x_train_list).shape}, y={np.array(y_train_list).shape}")

        # Print valid set shape
        x_valid_list, y_valid_list = [], []
        for (xx, yy) in valid_ds:
            x_valid_list.append(xx.numpy())
            y_valid_list.append(yy.item())
        print(f"Valid set shape: X={np.array(x_valid_list).shape}, y={np.array(y_valid_list).shape}")

        # Print test set shape
        x_test_list, y_test_list = [], []
        for (xx, yy) in test_ds:
            x_test_list.append(xx.numpy())
            y_test_list.append(yy.item())
        self.X_test = torch.from_numpy(np.array(x_test_list))
        self.y_true = torch.from_numpy(np.array(y_test_list))
        print(f"Test set shape: X={self.X_test.shape}, y={self.y_true.shape}")

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    def _eval_val(self, model):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in self.valid_loader:
                x, y = x.to(self.device).float(), y.to(self.device)
                out = model(x)
                preds.extend(torch.argmax(out, 1).cpu().numpy())
                labels.extend(y.cpu().numpy())
        acc = accuracy_score(labels, preds)
        f1 = self._binary_f1(labels, preds)
        return acc, f1

    def _train_loop(self, optimizer, criterion, epochs, patience):
        best_loss = float("inf")
        wait = 0
        self.best_model = copy.deepcopy(self.model)

        for ep in range(epochs):
            self.model.train()
            total_loss = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device).float(), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            avg_loss = total_loss / len(self.train_loader.dataset)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in self.valid_loader:
                    x, y = x.to(self.device).float(), y.to(self.device)
                    val_loss += criterion(self.model(x), y).item() * x.size(0)
            val_loss /= len(self.valid_loader.dataset)

            print(f"Epoch {ep+1}/{epochs} | Train Loss={avg_loss:.4f} | Val Loss={val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                self.best_model.load_state_dict(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break

        self.model.load_state_dict(self.best_model.state_dict())
        return best_loss

    def _optuna_objective(self, trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        wd = trial.suggest_float("wd", 1e-5, 1e-1, log=True)
        opt = trial.suggest_categorical("opt", ["adam", "sgd"])
        epochs = trial.suggest_int("epochs", 20, 60)

        self.preprocessing(normalize=self._normalize)
        self.data_loader(batch_size=32, balance=self._balance, balance_strategy=self._balance_strategy)

        if issubclass(self.model_class, PrototypeBasedModel):
            self.model = self.model_class(self.num_prototypes, self.n_vars, self.num_classes).to(self.device)
        else:
            self.model = self.model_class(self.window_size, self.n_vars, self.num_classes).to(self.device)

        self.model.apply(self._reset_weights)
        criterion = nn.CrossEntropyLoss()

        if opt == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd)

        patience = 10
        best_val_loss = float("inf")
        wait = 0

        for ep in range(epochs):
            self.model.train()
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()

            self.model.eval()
            vloss = 0
            with torch.no_grad():
                for x, y in self.valid_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    vloss += criterion(self.model(x), y).item() * x.size(0)
            vloss /= len(self.valid_loader.dataset)

            acc, f1 = self._eval_val(self.model)
            if self._opt_metric == "loss":
                metric = vloss
            elif self._opt_metric == "accuracy":
                metric = acc
            else:
                metric = f1

            trial.report(metric, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if vloss < best_val_loss:
                best_val_loss = vloss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        return metric

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_hpo: bool = False,
        n_trials: int = 30,
        optimize_metric: str = "loss",
        patience: int = 10,
        normalize: bool = True,
        balance: bool = False,
        balance_strategy: Literal["over", "under"] = "over",
    ):
        self._opt_metric = optimize_metric.lower()
        self._normalize = normalize
        self._balance = balance
        self._balance_strategy = balance_strategy

        if use_hpo:
            direction = "minimize" if self._opt_metric == "loss" else "maximize"
            study = optuna.create_study(direction=direction)
            study.optimize(self._optuna_objective, n_trials=n_trials)
            best = study.best_params
            with open(os.path.join(self.result_dir, "optuna_best_params.txt"), "w") as f:
                f.write(f"Best hyperparams:\n{best}\n")
                f.write(f"\nBest {self._opt_metric}: {study.best_value}\n")

            lr = best["lr"]
            wd = best["wd"]
            opt_method = best["opt"]
            ep = best["epochs"]

            self.preprocessing(normalize=normalize)
            self.data_loader(batch_size=batch_size, balance=balance, balance_strategy=balance_strategy)

            if issubclass(self.model_class, PrototypeBasedModel):
                self.model = self.model_class(self.num_prototypes, self.n_vars, self.num_classes).to(self.device)
            else:
                self.model = self.model_class(self.window_size, self.n_vars, self.num_classes).to(self.device)

            if opt_method == "adam":
                optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
            else:
                optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd)

            val_loss = self._train_loop(optimizer, nn.CrossEntropyLoss(), epochs=ep, patience=patience)
            return self.best_model, val_loss

        else:
            self.preprocessing(normalize=normalize)
            self.data_loader(batch_size=batch_size, balance=balance, balance_strategy=balance_strategy)

            if issubclass(self.model_class, PrototypeBasedModel):
                self.model = self.model_class(self.num_prototypes, self.n_vars, self.num_classes).to(self.device)
            else:
                self.model = self.model_class(self.window_size, self.n_vars, self.num_classes).to(self.device)

            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            val_loss = self._train_loop(optimizer, nn.CrossEntropyLoss(), epochs=epochs, patience=patience)
            return self.best_model, val_loss

    def evaluate(self):
        if self.test_loader is None:
            raise ValueError("Call data_loader() before evaluate().")
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device).float(), y.to(self.device)
                logits = self.model(x)
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        preds = np.array(preds)
        labels = np.array(labels)
        acc = accuracy_score(labels, preds)
        f1 = self._binary_f1(labels, preds)
        self.confusion_mat = confusion_matrix(labels, preds)
        print(f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}")

        plt.figure(figsize=(4, 3))
        sns.heatmap(self.confusion_mat, annot=True, cmap="Blues", fmt="d",
                    xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_path = os.path.join(self.result_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        return {"accuracy": acc, "f1": f1}



###############################################################################
# Example usage (main)
###############################################################################
if __name__ == "__main__":
    CSV_FILE_PATH = "../Dataset/ftse_minute_data_may_labelled.csv"
    n_var = 4
    baseline_models = [ResNet_baseline.ResNet, CNN_baseline.CNN, LSTM_baseline.LSTM,
                        MLP_baseline.MLP, FCN_baseline.FCN]
    for model_class in baseline_models:
        pipeline = Pipeline(
            model_class=model_class,
            file_path=CSV_FILE_PATH,
            n_vars=n_var,
            num_classes=2,
            result_dir=f"../Result/may/{model_class.__name__}",
            use_prototype=False
        )

        pipeline.train(
            use_hpo=True,
            epochs=10,
            batch_size=32,
            patience=5,
            normalize=True,
            balance=True,
            balance_strategy="over",
            optimize_metric="f1"
        )
        results = pipeline.evaluate()
        print("Baseline LSTM results:", results)

    # Prototype-based models
    selection_types = ["random", "k-means", "gmm"]
    distance_metrics = ["euclidean", "cosine"]
    for selection_type in selection_types:
        for distance_metric in distance_metrics:
            prototype_pipeline = Pipeline(
                model_class=PrototypeBasedModel,
                file_path=CSV_FILE_PATH,
                n_vars=n_var,
                num_classes=2,
                result_dir=f"../Result/may/PrototypeModel/{selection_type}_{distance_metric}",
                use_prototype=True,
                num_prototypes=10,
                prototype_selection_type=selection_type,
                prototype_distance_metric=distance_metric
            )

            prototype_pipeline.train(
                use_hpo=True,
                epochs=10,
                batch_size=32,
                patience=5,
                normalize=True,
                balance=True,
                balance_strategy="over",
                optimize_metric="f1"
            )
            proto_results = prototype_pipeline.evaluate()
            print("Prototype model results:", proto_results)
