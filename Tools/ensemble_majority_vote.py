# -*- coding: utf-8 -*-
"""Utility for ensembling prototype-based models via majority vote."""

from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np

from Pipeline.Pipeline import Pipeline, PrototypeModelBase


SELECTION_TYPES = ["random", "k-means", "gmm"]
DISTANCE_METRICS = ["euclidean", "cosine"]


def _train_and_predict(model_cls: type[PrototypeModelBase], sel: str, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """Train a single model and return its test probabilities and labels."""
    pipe = Pipeline(
        model_class=model_cls,
        file_path="../Dataset/ftse_minute_data_daily_labelled.csv",
        n_vars=5,
        num_classes=2,
        result_dir=f"../Result/ensemble/{model_cls.__name__}/{sel}_{metric}",
        use_prototype=True,
        num_prototypes=10,
        prototype_selection_type=sel,
        prototype_distance_metric=metric,
    )
    pipe.train(epochs=10, batch_size=32, balance=True, balance_strategy="over", normalize=True, use_hpo=True, n_trials=10)
    pipe.evaluate(threshold=0.5)
    probs, labels = pipe.predict_proba()
    return probs, labels


def majority_vote(pred_matrix: np.ndarray, vote_k: int) -> np.ndarray:
    """Apply majority vote with threshold ``vote_k``."""
    return (pred_matrix.sum(axis=0) >= vote_k).astype(int)


def ensemble_experiment(model_cls: type[PrototypeModelBase]) -> None:
    """Run ensemble experiment for a given prototype model class."""
    all_probs: List[np.ndarray] = []
    labels: np.ndarray | None = None
    for sel, metric in itertools.product(SELECTION_TYPES, DISTANCE_METRICS):
        probs, lbls = _train_and_predict(model_cls, sel, metric)
        if labels is None:
            labels = lbls
        all_probs.append(probs)
    pred_matrix = np.array([(p >= 0.5).astype(int) for p in all_probs])

    for k in range(1, pred_matrix.shape[0] + 1):
        final_preds = majority_vote(pred_matrix, k)
        f1 = Pipeline._binary_f1(labels, final_preds)
        print(f"Vote >= {k}: F1={f1:.4f}")


if __name__ == "__main__":
    # Example with PrototypeResNet as backbone
    from PrototypeBasedModel import PrototypeResNet

    ensemble_experiment(PrototypeResNet)
