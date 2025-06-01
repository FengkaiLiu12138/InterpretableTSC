import numpy as np
from Tools.DatasetConverter import DatasetConverter

DATA_PATH = 'Dataset/ftse_minute_data_daily.csv'
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


def simple_smote(X, y, random_state=42):
    rng = np.random.default_rng(random_state)
    maj_count = np.sum(y == 0)
    min_count = np.sum(y == 1)
    if min_count == 0:
        raise ValueError("No minority samples to oversample")
    n_needed = maj_count - min_count
    minority_idx = np.where(y == 1)[0]
    synth_samples = []
    for _ in range(n_needed):
        i = rng.choice(minority_idx)
        j = rng.choice(minority_idx)
        while j == i:
            j = rng.choice(minority_idx)
        diff = X[j] - X[i]
        gap = rng.random()
        new_sample = X[i] + gap * diff
        synth_samples.append(new_sample)
    X_syn = np.array(synth_samples)
    y_syn = np.ones(len(X_syn), dtype=y.dtype)
    X_bal = np.concatenate([X, X_syn], axis=0)
    y_bal = np.concatenate([y, y_syn], axis=0)
    return X_bal, y_bal, X_syn


def main():
    dc = DatasetConverter(file_path=DATA_PATH, save_path=None)
    df = dc.convert(label_type=1, volume=True)
    X, y = build_windows(df)
    X_bal, y_bal, X_syn = simple_smote(X, y)
    print(f"Original dataset: X={X.shape}, y={y.shape}")
    print(f"After SMOTE: X={X_bal.shape}, y={y_bal.shape}")
    np.set_printoptions(precision=4, suppress=True)
    for idx, win in enumerate(X_syn[:2], 1):
        print(f"\nSynthetic window {idx} (shape {win.shape}):")
        print(win)


if __name__ == '__main__':
    main()
