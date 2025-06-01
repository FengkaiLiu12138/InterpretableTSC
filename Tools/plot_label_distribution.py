import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Tools.DatasetConverter import DatasetConverter

DATASET_1 = os.path.join('..', 'Dataset', 'ftse_minute_data_may_labelled.csv')
DATASET_2 = os.path.join('..', 'Dataset', 'ftse_minute_data_daily_labelled.csv')


def _load_counts(csv_path: str) -> pd.Series:
    dc = DatasetConverter(file_path=csv_path, save_path=None)
    df = dc.convert(label_type=1, volume=True)
    return df['Labels'].value_counts().sort_index()


def plot_label_distribution(out_path: str = os.path.join('figures', 'label_distribution.png')) -> None:
    counts1 = _load_counts(DATASET_1)
    counts2 = _load_counts(DATASET_2)
    labels = sorted(set(counts1.index).union(counts2.index))

    values1 = counts1.reindex(labels, fill_value=0)
    values2 = counts2.reindex(labels, fill_value=0)

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar([i - width / 2 for i in x], values1, width=width, label='Dataset 1')
    plt.bar([i + width / 2 for i in x], values2, width=width, label='Dataset 2')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution Comparison')
    plt.xticks(list(x), labels)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == '__main__':
    plot_label_distribution()
