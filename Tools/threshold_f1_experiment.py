import os

from Pipeline.Pipeline import Pipeline
from BaselineModel import MLP_baseline
from Tools.DatasetConverter import DatasetConverter


RESULT_DIR = os.path.join('..', 'Result', 'threshold_f1')
RAW_CSV = os.path.join('..', 'Dataset', 'ftse_minute_data_daily.csv')
LABELLED_CSV = os.path.join('..', 'Dataset', 'ftse_minute_data_daily_labelled.csv')


def ensure_labelled(csv_path: str, labelled_path: str) -> str:
    """Generate labelled CSV if it does not exist."""
    if os.path.exists(labelled_path):
        return labelled_path
    dc = DatasetConverter(file_path=csv_path, save_path=labelled_path)
    dc.convert(label_type=1, volume=True)
    return labelled_path


def main() -> None:
    labelled = ensure_labelled(RAW_CSV, LABELLED_CSV)

    pipe = Pipeline(
        model_class=MLP_baseline.MLP,
        file_path=labelled,
        n_vars=5,
        num_classes=2,
        result_dir=RESULT_DIR,
        use_prototype=False,
    )

    pipe.train(
        epochs=5,
        batch_size=32,
        normalize=True,
        balance=True,
        balance_strategy='over',
        optimize_metric='f1',
    )

    best_th = pipe.find_best_threshold(step=0.01, metric='f1', plot_curve=True)
    print(f'Best threshold: {best_th:.2f}')

    pipe.evaluate(threshold=best_th)


if __name__ == '__main__':
    main()
