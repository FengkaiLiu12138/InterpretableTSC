import os
import itertools
from typing import List, Tuple

from Pipeline.Pipeline import Pipeline
from PrototypeBasedModel import PrototypeResNet
from Tools.DatasetConverter import DatasetConverter


PROTOTYPE_COUNTS = [10, 20, 30, 40, 50]
SELECTION_TYPES = ["random", "positive_only", "k-means", "gmm"]
DISTANCE_METRICS = ["euclidean", "cosine", "dtw"]


def ensure_labelled(csv_path: str, labelled_path: str) -> str:
    """Generate labelled CSV if it does not exist."""
    if os.path.exists(labelled_path):
        return labelled_path
    dc = DatasetConverter(file_path=csv_path, save_path=labelled_path)
    dc.convert(label_type=1, volume=True)
    return labelled_path


def run_single_experiment(num_proto: int, sel: str, metric: str, data_path: str) -> float:
    """Train and evaluate a prototype ResNet with given settings."""
    result_dir = f"../Result/prototype_number/{num_proto}/{sel}_{metric}"
    pipe = Pipeline(
        model_class=PrototypeResNet,
        file_path=data_path,
        n_vars=5,
        num_classes=2,
        result_dir=result_dir,
        use_prototype=True,
        num_prototypes=num_proto,
        prototype_selection_type=sel,
        prototype_distance_metric=metric,
    )
    pipe.train(use_hpo=True, n_trials=10, epochs=10, batch_size=32, balance=True,
               balance_strategy="over", normalize=True, optimize_metric="f1")
    best_th = pipe.find_best_threshold(step=0.01, metric="f1", plot_curve=False)
    results = pipe.evaluate(threshold=best_th)
    return results["f1"]


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "Dataset", "ftse_minute_data_daily.csv")
    labelled_path = os.path.join(base_dir, "..", "Dataset", "ftse_minute_data_daily_labelled.csv")
    data_path = ensure_labelled(csv_path, labelled_path)

    summary: List[Tuple[int, str, str, float]] = []
    for num_proto in PROTOTYPE_COUNTS:
        best_f1 = -1.0
        best_combo = ("", "")
        for sel, metric in itertools.product(SELECTION_TYPES, DISTANCE_METRICS):
            f1 = run_single_experiment(num_proto, sel, metric, data_path)
            summary.append((num_proto, sel, metric, f1))
            if f1 > best_f1:
                best_f1 = f1
                best_combo = (sel, metric)
        print(f"[Prototypes={num_proto}] Best combination: {best_combo[0]} + {best_combo[1]} => F1={best_f1:.4f}")

    print("\nFull summary:")
    for num_proto, sel, metric, f1 in summary:
        print(f"P{num_proto} | {sel} | {metric} | F1={f1:.4f}")


if __name__ == "__main__":
    main()
