import os
import sys
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch

from PrototypeBasedModel import PrototypeFeatureExtractor


DATA_PATH = "../Dataset/ftse_minute_data_daily.csv"


def _prepare_tensors(
    csv_path: str,
    window_size: int = 600,
    num_samples: int = 1,
    num_prototypes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load CSV and create sample windows and prototypes."""
    df = pd.read_csv(csv_path, header=None)
    values = df.iloc[:, 1:].values.astype("float32")

    segments = []
    for start in range(num_samples + num_prototypes):
        end = start + window_size
        if end > len(values):
            break
        segments.append(values[start:end])
    segments = torch.tensor(segments)
    samples = segments[:num_samples]
    protos = segments[num_samples : num_samples + num_prototypes]
    return samples, protos


def _time_metrics(
    series: torch.Tensor, protos: torch.Tensor, metrics: List[str]
) -> List[Tuple[str, float]]:
    """Measure time taken for each metric."""
    extractor = PrototypeFeatureExtractor(series, protos)
    results = []
    for m in metrics:
        start = time.perf_counter()
        extractor.compute_prototype_features(metric=m)
        elapsed = time.perf_counter() - start
        results.append((m, elapsed))
    return results


def main() -> None:
    metrics = ["euclidean", "cosine", "dtw"]
    ts, protos = _prepare_tensors(DATA_PATH)
    times = _time_metrics(ts, protos, metrics)

    df = pd.DataFrame(times, columns=["metric", "time_seconds"])
    df.to_csv("../Tools/metric_time_results.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(df["metric"], df["time_seconds"], color="skyblue")
    plt.ylabel("Time (s)")
    plt.title("Metric Computation Time")
    plt.tight_layout()
    plt.savefig("../Tools/metric_time_plot.png")
    plt.close()

    print(df)


if __name__ == "__main__":
    main()
