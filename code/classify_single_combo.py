"""Train and evaluate MUSA from saved features."""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from network.mua_common import load_fc_matrices, flatten_and_classify_mlp, set_global_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results"
CHANNEL_FILES = {
    "GM": "smri_GM_vit_features.npy",
    "WM": "smri_WM_vit_features.npy",
    "CSF": "smri_CSF_vit_features.npy",
    "ReHo": "fmri_ReHo_vit_features.npy",
    "DC": "fmri_DC_vit_features.npy",
    "ALFF": "fmri_ALFF_vit_features.npy",
    "VMHC": "fmri_VMHC_vit_features.npy",
}


def load_inputs(data_dir, feature_dir, fused_features, channels):
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

    sample_ids = np.arange(len(labels))
    file_list_path = data_dir / "file_list.npy"
    if file_list_path.is_file() or (fused_features is None and "FC" in channels):
        file_list = np.load(file_list_path, allow_pickle=True)
        if file_list.shape != labels.shape:
            raise ValueError("file_list.npy must contain one original filename per feature row.")
        sample_ids = [str(p).replace("\\", "/").split("/")[-1].removesuffix(".gz").removesuffix(".nii")
                      for p in file_list]

    if fused_features is not None:
        features = np.load(fused_features, mmap_mode="r")
        if (features.ndim not in (2, 3) or features.shape[0] != len(labels)
                or features.shape[-1] == 0 or (features.ndim == 3 and features.shape[1] != 116)
                or features.dtype.kind not in "fiu"):
            raise ValueError("Fused features must be numeric with shape [N,D] or [N,116,D].")
    else:
        arrays = []
        for ch in channels:
            values = (load_fc_matrices(data_dir, file_list) if ch == "FC"
                      else np.load(feature_dir / CHANNEL_FILES[ch], mmap_mode="r"))
            if (values.ndim != 3 or values.shape[:2] != (len(labels), 116)
                    or values.shape[2] == 0 or values.dtype.kind not in "fiu"):
                raise ValueError(f"{ch} features must be numeric with shape [N,116,D].")
            arrays.append(values)
        features = np.concatenate(arrays, axis=2)
    return features, labels, train, test, sample_ids, int(split["random_state"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, help="Fused .npy file. Overrides channel cache loading.")
    parser.add_argument("--feature-dir", type=Path, default=PROJECT_ROOT / "3dvit_features")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--channels", nargs="+", choices=[*CHANNEL_FILES, "FC"],
                        default=["WM", "FC", "ReHo", "DC"])
    args = parser.parse_args()
    if len(set(args.channels)) != len(args.channels):
        parser.error("--channels must not contain duplicates")
    run_classification_experiment(args.channels, args.data_dir, args.feature_dir, args.features)


def run_classification_experiment(selected_channels, data_dir, feature_dir, fused_features=None):
    concatenated, labels, train_indices, test_indices, sample_ids, random_state = load_inputs(
        data_dir, feature_dir, fused_features, selected_channels
    )
    combination_name = "Fused" if fused_features is not None else "+".join(selected_channels)
    selected_channels = None if fused_features is not None else selected_channels
    cv_folds = 5
    if np.min(np.bincount(labels[train_indices].astype(int))) < cv_folds:
        raise ValueError("The training set needs at least five samples per class for 5-fold CV.")
    set_global_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Feature shape: {concatenated.shape}, device: {device}, seed: {random_state}")
    predictions = []

    # 4. Inner cross-validation classification
    print("\n" + "=" * 80)
    print("Starting inner %d-fold cross-validation" % (cv_folds))
    print("=" * 80)
    print("Train-validation samples: %d" % (len(train_indices)))
    print("Holdout test samples: %d" % (len(test_indices)))
    
    train_labels = labels[train_indices]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    fold_results = []
    
    for fold, (train_subset_idx, val_subset_idx) in enumerate(
        skf.split(train_indices, train_labels), 1
    ):
        print("\n" + "=" * 80)
        print("Fold %d/%d" % (fold, cv_folds))
        print("=" * 80)
        
        train_subset = train_indices[train_subset_idx]  # 4-fold training
        val_subset = train_indices[val_subset_idx]  # 1-fold validation (early stopping)
        
        print("  Training subset: %d samples" % (len(train_subset)))
        print("  Validation subset: %d samples" % (len(val_subset)))
        print("  Holdout test set: %d samples" % (len(test_indices)))
        
        _, accuracy, f1, auc, metrics = flatten_and_classify_mlp(
            concatenated,
            labels,
            train_subset,
            test_indices,
            method_name=f"{combination_name}-fold{fold}",
            device=device,
            val_indices=val_subset,
            use_multi_checkpoint=True,
            checkpoint_interval=10,
            num_epochs=400,
            early_stop_patience=100,
            enable_early_stop=True,
        )

        predictions.extend(
            (fold, idx, sample_ids[idx], int(labels[idx]), prediction, probability)
            for idx, prediction, probability in zip(
                test_indices, metrics["predicted_labels"], metrics["mdd_probabilities"]
            )
        )
        sensitivity = metrics.get("sensitivity")
        specificity = metrics.get("specificity")
        precision = metrics.get("precision")
        confusion_matrix = metrics.get("confusion_matrix")

        print(
            "  Fold %d: ACC=%.4f, SEN=%.4f, SPE=%.4f, PRE=%.4f, F1=%.4f, AUC=%.4f"
            % (
                fold,
                float(accuracy),
                float(sensitivity) if sensitivity is not None else -1.0,
                float(specificity) if specificity is not None else -1.0,
                float(precision) if precision is not None else -1.0,
                float(f1),
                float(auc) if auc is not None else -1.0,
            )
        )

        fold_results.append(
            {
                "fold": fold,
                "accuracy": float(accuracy),
                "sensitivity": float(sensitivity) if sensitivity is not None else None,
                "specificity": float(specificity) if specificity is not None else None,
                "precision": float(precision) if precision is not None else None,
                "f1": float(f1),
                "auc": float(auc) if auc is not None else None,
                "confusion_matrix": (
                    confusion_matrix.tolist()
                    if confusion_matrix is not None and hasattr(confusion_matrix, 'tolist') and not isinstance(confusion_matrix, (list, tuple))
                    else confusion_matrix if confusion_matrix is not None
                    else None
                ),
            }
        )
    
    # 5. Summarize results
    print("\n" + "=" * 80)
    print("Experiment Results Summary")
    print("=" * 80)
    
    valid_results = [r for r in fold_results if r.get("accuracy") is not None]
    if valid_results:
        accs = np.array([r["accuracy"] for r in valid_results], dtype=float)
        f1s = np.array([r["f1"] for r in valid_results], dtype=float)
        aucs = np.array(
            [r["auc"] for r in valid_results if r.get("auc") is not None], dtype=float
        )
        
        sens = np.array(
            [r["sensitivity"] for r in valid_results if r.get("sensitivity") is not None],
            dtype=float,
        )
        spes = np.array(
            [r["specificity"] for r in valid_results if r.get("specificity") is not None],
            dtype=float,
        )
        pres = np.array(
            [r["precision"] for r in valid_results if r.get("precision") is not None],
            dtype=float,
        )
        
        def mean_std(arr):
            if arr.size == 0:
                return None, None
            return float(np.mean(arr)), float(np.std(arr, ddof=1))
        
        acc_mean, acc_std = mean_std(accs)
        f1_mean, f1_std = mean_std(f1s)
        auc_mean, auc_std = mean_std(aucs)
        sen_mean, sen_std = mean_std(sens)
        spe_mean, spe_std = mean_std(spes)
        pre_mean, pre_std = mean_std(pres)
        
        print("ACC: %.4f +/- %.4f (%.2f%% +/- %.2f%%)" % (acc_mean,
            acc_std,
            acc_mean * 100.0,
            acc_std * 100.0))
        if sen_mean is not None:
            print("SEN: %.4f +/- %.4f" % (sen_mean, sen_std))
        if spe_mean is not None:
            print("SPE: %.4f +/- %.4f" % (spe_mean, spe_std))
        if pre_mean is not None:
            print("PRE: %.4f +/- %.4f" % (pre_mean, pre_std))
        if f1_mean is not None:
            print("F1 : %.4f +/- %.4f" % (f1_mean, f1_std))
        if auc_mean is not None:
            print("AUC: %.4f +/- %.4f" % (auc_mean, auc_std))
        
        summary_metrics = {
            "accuracy": {"mean": acc_mean, "std": acc_std},
            "f1": {"mean": f1_mean, "std": f1_std} if f1_mean is not None else None,
            "auc": {"mean": auc_mean, "std": auc_std} if auc_mean is not None else None,
            "sensitivity": {"mean": sen_mean, "std": sen_std} if sen_mean is not None else None,
            "specificity": {"mean": spe_mean, "std": spe_std} if spe_mean is not None else None,
            "precision": {"mean": pre_mean, "std": pre_std} if pre_mean is not None else None,
        }
    else:
        print("All folds failed; cannot compute summary metrics.")
        summary_metrics = None
    
    # 6. Save results to JSON
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_file = os.path.join(
        RESULT_DIR,
        f"classification_{combination_name}_seed{random_state}_inner{cv_folds}fold.json",
    )
    
    output_data = {
        "experiment_name": f"Classification: {combination_name}",
        "combination_name": combination_name,
        "channels": selected_channels,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": str(fused_features) if fused_features is not None else None,
        "fold_results": fold_results,
        "summary_metrics": summary_metrics,
    }
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    prediction_file = Path(result_file).with_suffix(".csv")
    with prediction_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "row_index", "sample_id", "true_label", "predicted_label", "mdd_probability"])
        writer.writerows(predictions)

    print(f"Predictions saved to: {prediction_file}")
    print("\nExperiment results saved to: %s" % (result_file))
    print("Experiment completed, end time: %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
    
    return output_data


if __name__ == '__main__':
    main()
