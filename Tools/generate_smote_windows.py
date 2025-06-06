import os
import numpy as np
import matplotlib.pyplot as plt
from Tools.DatasetConverter import DatasetConverter

try:
    from imblearn.over_sampling import SMOTE
    _has_smote = True
except ImportError:
    _has_smote = False
    print("Warning: imbalanced-learn is not installed. SMOTE will not be available.")

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'ftse_minute_data_daily.csv')

WINDOW_SIZE = 600
FEATURE_COLS = ["Close", "High", "Low", "Open", "Volume"]


def build_windows(df):
    half = WINDOW_SIZE // 2
    X_list, y_list = [], []
    for start in range(len(df) - WINDOW_SIZE + 1):
        end = start + WINDOW_SIZE
        window = df.iloc[start:end][FEATURE_COLS].values
        label = df.iloc[start + half]["Labels"]
        X_list.append(window)
        y_list.append(label)
    return np.array(X_list), np.array(y_list)


def pipeline_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """Apply SMOTE using the same approach as ``Pipeline.data_loader``."""
    if not _has_smote:
        raise ImportError(
            "imbalanced-learn is not installed. SMOTE is unavailable."
        )

    N, W, D = X.shape
    X_2d = X.reshape(N, -1)
    sm = SMOTE(random_state=random_state)
    X_bal_2d, y_bal = sm.fit_resample(X_2d, y)
    new_N = X_bal_2d.shape[0]
    X_bal = X_bal_2d.reshape(new_N, W, D)
    X_syn = X_bal[len(X) :]
    return X_bal, y_bal, X_syn


def plot_window(window: np.ndarray, idx: int, out_dir: str = 'figures') -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 3))
    for col_idx, name in enumerate(FEATURE_COLS):
        plt.plot(window[:, col_idx], label=name)
    plt.axvline(WINDOW_SIZE // 2, color='r', ls=':')
    plt.xlabel('Time Step')
    plt.title(f'SMOTE Window {idx}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'smote_window_{idx}.png'))
    plt.close()


def main():
    dc = DatasetConverter(file_path=DATA_PATH, save_path=None)
    df = dc.convert(label_type=1, volume=True)
    X, y = build_windows(df)
    X_bal, y_bal, X_syn = pipeline_smote(X, y)
    print(f"Original dataset: X={X.shape}, y={y.shape}")
    print(f"After SMOTE: X={X_bal.shape}, y={y_bal.shape}")
    np.set_printoptions(precision=4, suppress=True)
    for idx, win in enumerate(X_syn[:2], 1):
        print(f"\nSynthetic window {idx} (shape {win.shape}):")
        print(win)
        plot_window(win, idx)

if __name__ == '__main__':
    main()
