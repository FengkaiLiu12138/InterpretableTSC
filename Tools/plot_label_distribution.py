import os
import pandas as pd
import matplotlib.pyplot as plt


# Default labelled datasets used for demonstration. These paths can be
# overridden via command line arguments.
DATASET_1 = os.path.join('..', 'Dataset', 'ftse_minute_data_may_labelled.csv')
DATASET_2 = os.path.join('..', 'Dataset', 'ftse_minute_data_daily_labelled.csv')


def _load_counts(csv_path: str) -> pd.Series:
    """Load label counts from a labelled CSV file.

    The function expects a column named ``Labels``. Only this column is
    required, making the loader resilient to different feature sets.
    """

    df = pd.read_csv(csv_path)
    if "Labels" not in df.columns:
        raise ValueError(f"Expected column 'Labels' in {csv_path}")
    return df["Labels"].value_counts().sort_index()


def plot_label_distribution(
    dataset_1: str = DATASET_1,
    dataset_2: str = DATASET_2,
    out_path: str = os.path.join("figures", "label_distribution.png"),
) -> None:
    """Plot label count comparison for two labelled datasets."""

    counts1 = _load_counts(dataset_1)
    counts2 = _load_counts(dataset_2)
    labels = sorted(set(counts1.index).union(counts2.index))

    values1 = counts1.reindex(labels, fill_value=0)
    values2 = counts2.reindex(labels, fill_value=0)

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar([i - width / 2 for i in x], values1, width=width, label='Minute')
    plt.bar([i + width / 2 for i in x], values2, width=width, label='Daily')
    for i, v in enumerate(values1):
        plt.text(i - width / 2, v + 0.05, str(v), ha='center', va='bottom')
    for i, v in enumerate(values2):
        plt.text(i + width / 2, v + 0.05, str(v), ha='center', va='bottom')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution for Both Datasets')
    plt.xticks(list(x), labels)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == '__main__':
    plot_label_distribution()