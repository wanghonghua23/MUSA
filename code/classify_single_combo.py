"""Run the original MUSA MLP on one fused feature array and report test ACC."""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from network.mua_common import flatten_and_classify_mlp, set_global_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CV_FOLDS = 5


def load_inputs(data_dir, features_path):
    labels = np.load(data_dir / "labels.npy")
    # Original split cache stores a pickled dict. Use only trusted project files.
    split = np.load(data_dir / "data_split.npy", allow_pickle=True).item()
    train, test = split["train_indices"], split["test_indices"]
    if labels.ndim != 1 or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("labels.npy must contain one binary label per sample (HC=0, MDD=1).")
    if any(idx.ndim != 1 or idx.dtype.kind not in "iu" for idx in (train, test)):
        raise ValueError("Split indices must be one-dimensional integer arrays.")
    if not np.array_equal(np.sort(np.concatenate((train, test))), np.arange(len(labels))):
        raise ValueError("Train/test split must partition all feature rows without overlap.")
    if any(np.unique(labels[idx]).size != 2 for idx in (train, test)):
        raise ValueError("Both train and test must contain HC and MDD samples.")

    features = np.load(features_path, mmap_mode="r")
    if (features.ndim not in (2, 3) or features.shape[0] != len(labels)
            or features.shape[-1] == 0 or (features.ndim == 3 and features.shape[1] != 116)
            or features.dtype.kind not in "fiu"):
        raise ValueError("Expected numeric features with shape [N,D] or [N,116,D].")
    return features, labels, train, test, int(split["random_state"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Fused feature .npy file")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()

    features, labels, train, test, seed = load_inputs(args.data_dir, args.features)
    if np.bincount(labels[train].astype(int), minlength=2).min() < CV_FOLDS:
        raise ValueError("The training set needs at least five samples per class.")
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    scores = []
    for train_rows, val_rows in cv.split(train, labels[train]):
        accuracy = flatten_and_classify_mlp(
            features, labels, train[train_rows], test, device=device,
            val_indices=train[val_rows], use_multi_checkpoint=True,
        )
        scores.append(accuracy)

    print(f"ACC: {np.mean(scores):.2%} +/- {np.std(scores, ddof=1):.2%}")


if __name__ == "__main__":
    main()
