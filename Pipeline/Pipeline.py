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

# ============ Baseline Models (示例) ============
# 这里只是示例导入，实际需确认你项目结构中对应模块路径
from BaselineModel import (
    ResNet_baseline,
    CNN_baseline,
    LSTM_baseline,
    MLP_baseline,
    FCN_baseline
)

# ============ Prototype Models (示例) ============
# 这里只是示例导入，实际需确认你项目结构中对应模块路径
from PrototypeBasedModel import PrototypeBasedModel
from PrototypeBasedModel.PrototypeBasedModel import PrototypeBasedModel, PrototypeFeatureExtractor, PrototypeSelector


###############################################################################
#  Pipeline
###############################################################################
class Pipeline:
    """
    End-to-end pipeline with optional class balancing and Optuna HPO.
    * Default: binary classification, labels {0, 1} with 1 as the positive class
    * F1 is computed with pos_label=1
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
            prototype_distance_metric: str = 'euclidean'
    ):
        """
        Args:
            model_class: The model class (e.g. ResNet_baseline.ResNet, or PrototypeBasedModel)
            file_path: Path to your CSV
            n_vars: Number of features
            num_classes: Number of classes (2 for binary)
            result_dir: Where to store figures & results
            use_prototype: 是否启用原型模式
            num_prototypes: 原型个数
            prototype_selection_type: 原型选择方式 ('random', 'k-means', 'gmm')
            prototype_distance_metric: 距离度量 ('euclidean', 'dtw', 'cosine', 'mahalanobis')
        """
        self.model_class = model_class
        self.file_path = file_path
        self.n_vars = n_vars
        self.num_classes = num_classes
        self.window_size = 600
        self.model = None

        model_type = model_class.__name__
        if result_dir is None:
            result_dir = f"../Result/{model_type}"
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} | result_dir={self.result_dir}")

        self._df = None
        self.dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.best_model = None
        self.X_test = None
        self.y_true = None
        self.confusion_mat = None
        self._opt_metric = "loss"

        self.use_prototype = use_prototype
        self.num_prototypes = num_prototypes
        self.prototype_selection_type = prototype_selection_type
        self.prototype_distance_metric = prototype_distance_metric

        self._balance = False
        self._balance_strategy = "over"
        self._normalize = True

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

    def preprocessing(
            self,
            *,
            normalize: bool = True,
            balance: bool = False,
            balance_strategy: Literal["over", "under"] = "over",
    ) -> None:
        self._normalize = normalize
        self._balance = balance
        self._balance_strategy = balance_strategy

        df = self._df.copy()
        feature_cols = ["Close", "High", "Low", "Open", "Volume"][: self.n_vars]

        if normalize:
            for col in feature_cols:
                minv = df[col].min()
                maxv = df[col].max()
                df[col] = (df[col] - minv) / (maxv - minv + 1e-12)

        X_list = []
        y_list = []
        half_w = self.window_size // 2
        total = len(df)

        for start_idx in range(0, total - self.window_size + 1):
            end_idx = start_idx + self.window_size
            window_data = df.iloc[start_idx:end_idx][feature_cols].values
            center_label_idx = start_idx + half_w
            label = df.iloc[center_label_idx]["Labels"]
            X_list.append(window_data)
            y_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        if balance:
            count_0 = np.sum(y == 0)
            count_1 = np.sum(y == 1)
            if count_0 > count_1:
                maj_label, min_label = 0, 1
            else:
                maj_label, min_label = 1, 0

            X_maj = X[y == maj_label]
            X_min = X[y == min_label]

            if balance_strategy == "over":
                X_min_resampled = resample(X_min, replace=True, n_samples=len(X_maj), random_state=42)
                y_min_resampled = np.array([min_label] * len(X_maj), dtype=np.int64)
                X = np.concatenate([X_maj, X_min_resampled], axis=0)
                y = np.concatenate([
                    np.full(len(X_maj), maj_label, dtype=np.int64),
                    y_min_resampled
                ], axis=0)
            elif balance_strategy == "under":
                X_maj_resampled = resample(X_maj, replace=False, n_samples=len(X_min), random_state=42)
                y_maj_resampled = np.array([maj_label] * len(X_min), dtype=np.int64)
                X = np.concatenate([X_maj_resampled, X_min], axis=0)
                y = np.concatenate([
                    y_maj_resampled,
                    np.full(len(X_min), min_label, dtype=np.int64)
                ], axis=0)

            shuffle_idx = np.random.RandomState(42).permutation(len(X))
            X = X[shuffle_idx]
            y = y[shuffle_idx]
            print("Balanced labels:", np.unique(y, return_counts=True))

        if self.use_prototype:
            print(f"[Prototype Mode] Selecting {self.num_prototypes} prototypes via '{self.prototype_selection_type}' ...")
            selector = PrototypeSelector(X, y, window_size=self.window_size)
            prototypes, proto_labels, remaining_data, remaining_labels = selector.select_prototypes(
                num_prototypes=self.num_prototypes,
                selection_type=self.prototype_selection_type
            )
            print(f"Prototypes shape = {prototypes.shape}, labels = {proto_labels.shape}")
            print(f"Remaining shape  = {remaining_data.shape}, labels = {remaining_labels.shape}")

            time_series_t = torch.from_numpy(remaining_data)
            prototypes_t = torch.from_numpy(prototypes)

            extractor = PrototypeFeatureExtractor(time_series_t, prototypes_t)
            extractor.plot_prototype_feature_map(metric=self.prototype_distance_metric,
                                                 save_path=os.path.join(self.result_dir, "prototype_feature_map.png"))

            feats_t = extractor.compute_prototype_features(metric=self.prototype_distance_metric)
            X = feats_t.numpy()
            y = remaining_labels

            print(f"[Prototype Mode] After feature extraction, X shape = {X.shape}, y shape = {y.shape}")

        self.dataset = (X, y)

    def data_loader(
            self,
            *,
            batch_size: int = 32,
            train_ratio: float = 0.7,
            valid_ratio: float = 0.15,
            test_ratio: float = 0.15,
    ):
        if self.dataset is None:
            raise ValueError("Call preprocessing() first")

        X, y = self.dataset
        N = len(X)

        if self.use_prototype:
            print(f"[data_loader] Full dataset shape = {X.shape} (N, {self.num_prototypes}, {self.n_vars})")
        else:
            print(f"[data_loader] Full dataset shape = {X.shape} (N, {self.window_size}, {self.n_vars})")

        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()

        dataset = TensorDataset(X_tensor, y_tensor)
        total = len(dataset)
        tr = int(train_ratio * total)
        va = int(valid_ratio * total)
        te = total - tr - va

        train_ds, valid_ds, test_ds = random_split(
            dataset, [tr, va, te], generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        self.X_test = torch.stack([dt[0] for dt in test_ds])
        self.y_true = torch.stack([dt[1] for dt in test_ds])

        print(f"Train DS size: {tr}")
        print(f"Valid DS size: {va}")
        print(f"Test  DS size: {te}")

        if not self.use_prototype:
            trainX = torch.stack([dt[0] for dt in train_ds])
            trainY = torch.stack([dt[1] for dt in train_ds])
            self._plot_example_sample(trainX, trainY, label=0, title="Example: label=0 (train)")
            self._plot_example_sample(trainX, trainY, label=1, title="Example: label=1 (train)")

    def _plot_example_sample(self, X_tensor, Y_tensor, label: int, title: str):
        idxs = (Y_tensor == label).nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
            print(f"No sample with label={label} found.")
            return
        sample_idx = idxs[0].item()
        window = X_tensor[sample_idx].cpu().numpy()
        center_idx = window.shape[0] // 2
        self._plot_window(window, center_idx, title=title, save_filename=None)

    def _plot_window(self, window: np.ndarray, center_idx: int, title: str, save_filename: str | None):
        T, C = window.shape
        x_vals = np.arange(T)

        plt.figure(figsize=(6, 4))
        for i in range(C):
            plt.plot(x_vals, window[:, i], label=f"Feature {i}")
        plt.scatter(center_idx, window[center_idx, 0], color='red', zorder=5, label='Center')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        if save_filename is not None:
            plt.savefig(os.path.join(self.result_dir, save_filename))
        plt.close()

    def _eval_val(self, model):
        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for x, y in self.valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                labels.extend(y.cpu().numpy())
        preds = np.array(preds)
        labels = np.array(labels)
        acc = accuracy_score(labels, preds)
        f1 = self._binary_f1(labels, preds)
        return acc, f1

    def _train_loop(self, optimizer, criterion, epochs, patience):
        best_loss = float("inf")
        wait = 0
        self.best_model = copy.deepcopy(self.model)

        for ep in range(epochs):
            self.model.train()
            total_loss = 0.0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)

            avg_loss = total_loss / len(self.train_loader.dataset)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in self.valid_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    val_loss += criterion(self.model(x), y).item() * x.size(0)
            val_loss /= len(self.valid_loader.dataset)

            if val_loss < best_loss:
                best_loss = val_loss
                self.best_model.load_state_dict(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self.model.load_state_dict(self.best_model.state_dict())
        return best_loss

    def _optuna_objective(self, trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        wd = trial.suggest_float("wd", 1e-5, 1e-1, log=True)
        opt = trial.suggest_categorical("opt", ["adam", "sgd"])
        epochs = trial.suggest_int("epochs", 20, 60)

        self.preprocessing(normalize=self._normalize, balance=self._balance, balance_strategy=self._balance_strategy)
        self.data_loader(batch_size=32)

        if issubclass(self.model_class, PrototypeBasedModel):
            self.model = self.model_class(
                num_prototypes=self.num_prototypes,
                n_var=self.n_vars,
                num_classes=self.num_classes
            ).to(self.device)
        else:
            self.model = self.model_class(
                window_size=self.window_size,
                n_vars=self.n_vars,
                num_classes=self.num_classes
            ).to(self.device)

        self.model.apply(self._reset_weights)
        criterion = nn.CrossEntropyLoss()
        optimizer = (optim.Adam if opt == "adam" else optim.SGD)(
            self.model.parameters(), lr=lr, weight_decay=wd
        )
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
            vloss = 0.0
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
            *,
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
            print("Optuna best params:", best)

            with open(os.path.join(self.result_dir, "optuna_best_params.txt"), "w") as f:
                f.write(f"Best hyperparams:\n{best}\n")
                f.write(f"\nBest {self._opt_metric}: {study.best_value}\n")

            lr = best["lr"]
            wd = best["wd"]
            opt = best["opt"]
            ep = best["epochs"]

            self.preprocessing(normalize=normalize, balance=balance, balance_strategy=balance_strategy)
            self.data_loader(batch_size=batch_size)

            if issubclass(self.model_class, PrototypeBasedModel):
                self.model = self.model_class(
                    num_prototypes=self.num_prototypes,
                    n_var=self.n_vars,
                    num_classes=self.num_classes
                ).to(self.device)
            else:
                self.model = self.model_class(
                    window_size=self.window_size,
                    n_vars=self.n_vars,
                    num_classes=self.num_classes
                ).to(self.device)

            optimizer = (optim.Adam if opt == "adam" else optim.SGD)(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
            val_loss = self._train_loop(optimizer, nn.CrossEntropyLoss(), epochs=ep, patience=patience)
            return self.best_model, val_loss
        else:
            self.preprocessing(normalize=normalize, balance=balance, balance_strategy=balance_strategy)
            self.data_loader(batch_size=batch_size)

            if issubclass(self.model_class, PrototypeBasedModel):
                self.model = self.model_class(
                    num_prototypes=self.num_prototypes,
                    n_var=self.n_vars,
                    num_classes=self.num_classes
                ).to(self.device)
            else:
                self.model = self.model_class(
                    window_size=self.window_size,
                    n_vars=self.n_vars,
                    num_classes=self.num_classes
                ).to(self.device)

            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            val_loss = self._train_loop(optimizer, nn.CrossEntropyLoss(), epochs=epochs, patience=patience)
            return self.best_model, val_loss

    def evaluate(self):
        if self.test_loader is None:
            raise ValueError("Must call data_loader() before evaluate().")

        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        preds = np.array(preds)
        labels = np.array(labels)
        acc = accuracy_score(labels, preds)
        f1 = self._binary_f1(labels, preds)

        self.confusion_mat = confusion_matrix(labels, preds)
        print(f"Test Accuracy: {acc:.4f} | F1 (pos=1): {f1:.4f}")

        plt.figure(figsize=(4, 3))
        sns.heatmap(self.confusion_mat, annot=True, cmap="Blues", fmt="d",
                    xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(self.result_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")

        if not self.use_prototype:
            X_test_np = self.X_test.cpu().numpy()
            tn_idx = np.where((labels == 0) & (preds == 0))[0]
            fp_idx = np.where((labels == 0) & (preds == 1))[0]
            fn_idx = np.where((labels == 1) & (preds == 0))[0]
            tp_idx = np.where((labels == 1) & (preds == 1))[0]

            cell_mapping = {
                "TN": tn_idx,
                "FP": fp_idx,
                "FN": fn_idx,
                "TP": tp_idx
            }

            for name, arr in cell_mapping.items():
                if len(arr) > 0:
                    ex_index = arr[0]
                    window = X_test_np[ex_index]
                    center_idx = window.shape[0] // 2
                    title = f"Example: {name} (true={labels[ex_index]}, pred={preds[ex_index]})"
                    filename = f"example_{name}.png"
                    self._plot_window(window, center_idx, title, save_filename=filename)
                    print(f"{name} example saved to: {os.path.join(self.result_dir, filename)}")
                else:
                    print(f"No example for {name} in test set.")
        else:
            print("[Prototype Mode] Skipping example window plots, because input is no longer raw time-series.")

            # 在原型模式下，如果模型是 PrototypeBasedModel，可以分别生成 TP, FP, FN, TN 的特征图
            if isinstance(self.model, PrototypeBasedModel) and self.X_test is not None:
                tn_idx = np.where((labels == 0) & (preds == 0))[0]
                fp_idx = np.where((labels == 0) & (preds == 1))[0]
                fn_idx = np.where((labels == 1) & (preds == 0))[0]
                tp_idx = np.where((labels == 1) & (preds == 1))[0]

                cat_indices = {
                    "TN": tn_idx,
                    "FP": fp_idx,
                    "FN": fn_idx,
                    "TP": tp_idx
                }
                for cat_name, idx_array in cat_indices.items():
                    if len(idx_array) > 0:
                        ex_index = idx_array[0]
                        x_sample = self.X_test[ex_index].unsqueeze(0).to(self.device)
                        block_outs = self.model.forward_with_intermediate(x_sample)
                        for i, bout in enumerate(block_outs, 1):
                            arr = bout[0].cpu().numpy()
                            plt.figure(figsize=(6, 4))
                            sns.heatmap(arr, cmap="viridis")
                            plt.title(f"{cat_name} Block {i} Feature Map")
                            plt.xlabel("Length")
                            plt.ylabel("Channels")
                            plt.tight_layout()
                            save_name = f"{cat_name.lower()}_block{i}_feature_map.png"
                            plt.savefig(os.path.join(self.result_dir, save_name))
                            plt.close()
                            print(f"{cat_name} Block {i} feature map saved to: {os.path.join(self.result_dir, save_name)}")
                    else:
                        print(f"[Prototype Mode] No example for {cat_name} in test set.")

        return {"accuracy": acc, "f1": f1}


###############################################################################
# Example usage (main)
###############################################################################
if __name__ == "__main__":
    # 1) Update your CSV path
    CSV_FILE_PATH = "../Dataset/ftse_minute_data_may_labelled.csv"
    n_var = 4
    # # ========== 示例1: 使用基线模型 (e.g. LSTM) ==========\
    baseline_models = [ResNet_baseline.ResNet, CNN_baseline.CNN, LSTM_baseline.LSTM,
                        MLP_baseline.MLP, FCN_baseline.FCN]
    for model_class in baseline_models:
        pipeline = Pipeline(
            model_class=model_class,
            file_path=CSV_FILE_PATH,
            n_vars=n_var,  # columns: Close, High, Low, Open, Volume
            num_classes=2,  # binary classification
            result_dir=f"../Result/may/{model_class.__name__}",
            use_prototype=False  # 关闭原型模式
        )

        pipeline.train(
            use_hpo=True,          # 不使用Optuna
            epochs=10,              # 只跑10个epoch做演示
            batch_size=32,
            patience=5,
            normalize=True,
            balance=True,
            balance_strategy="over",
            optimize_metric="f1"
        )
        results = pipeline.evaluate()
        print("Baseline LSTM results:", results)

    # ========== 示例2: 使用 PrototypeBasedModel ==========
    selection_types = ["random", "k-means", "gmm"]
    distance_metrics = ["euclidean", "cosine"]
    for selection_type in selection_types:
        for distance_metric in distance_metrics:
            prototype_pipeline = Pipeline(
                model_class=PrototypeBasedModel,  # 你的自定义原型模型
                file_path=CSV_FILE_PATH,
                n_vars=n_var,
                num_classes=2,
                result_dir=f"../Result/may/PrototypeModel/{selection_type}_{distance_metric}",
                use_prototype=True,  # 启用原型模式
                num_prototypes=10,  # 原型个数
                prototype_selection_type=selection_type,  # 原型选择方式
                prototype_distance_metric=distance_metric
            )

            prototype_pipeline.train(
                use_hpo=True,  # 同样可以开启 HPO
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
