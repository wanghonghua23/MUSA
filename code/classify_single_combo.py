#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MUSA single-combination classification

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

import json
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from data_split_and_paths import SCRIPT_DIR, FEATURE_DIR, RESULT_DIR, load_saved_data_split_only
from network.mua_common import (
    load_fc_matrices,
    flatten_and_classify_mlp,
    set_global_seed,
)
from extract_roi_vit_features import (
    extract_smri_all_channels_vit_features,
    extract_fmri_all_channels_vit_features,
    fmri_vit_features_cache_ready,
)

# Default experiment hyperparameters
default_channels                 = ["WM","FC","ReHo","DC"]
default_combination_name         = "WM+FC+ReHo+DC"
default_random_state             = 42
default_cv_folds                 = 5
default_force_recompute_features = False
holdout_test_size                = 0.2


def main():
    print("=" * 70)
    print("MUSA Single-Combination Classification")
    print("=" * 70)

    run_classification_experiment(
        selected_channels=list(default_channels),
        combination_name=default_combination_name,
        random_state=default_random_state,
        cv_folds=default_cv_folds,
        force_recompute_features=default_force_recompute_features,
    )


# Helper functions

def concatenate_selected_channels(
    smri_channels_dict,
    fc_matrices,
    fmri_channels_dict,
    selected_channels,
    include_fc=True,
):
    # Concatenate sMRI / FC / fMRI features along last dim in selected_channels order -> [N,116,D]
    if fc_matrices is None and "FC" in selected_channels and include_fc:
        raise ValueError("selected_channels includes 'FC' but fc_matrices is None")
    
    feature_list = []
    channel_info = []
    
    for ch_name in selected_channels:
        if smri_channels_dict is not None and ch_name in smri_channels_dict:
            feature_list.append(smri_channels_dict[ch_name])
            channel_info.append(f"{ch_name}({smri_channels_dict[ch_name].shape[2]})")
        elif fmri_channels_dict is not None and ch_name in fmri_channels_dict:
            feature_list.append(fmri_channels_dict[ch_name])
            channel_info.append(f"{ch_name}({fmri_channels_dict[ch_name].shape[2]})")
        elif ch_name == "FC" and include_fc and fc_matrices is not None:
            feature_list.append(fc_matrices)
            channel_info.append(f"FC({fc_matrices.shape[2]})")
        else:
            raise ValueError(f"No features found for channel {ch_name}")
    
    if not feature_list:
        raise ValueError(f"No valid channels found: {selected_channels}")
    
    concatenated = np.concatenate(feature_list, axis=2)
    print(
        "  Concatenated channels: %s => per_roi_dim = %d"
        % (" + ".join(channel_info), concatenated.shape[2])
    )
    
    return concatenated


def smri_vit_features_cache_ready(force_recompute=False):
    # Whether sMRI three-channel and metadata cache are complete
    if force_recompute:
        return False
    required = (
        "smri_GM_vit_features.npy",
        "smri_WM_vit_features.npy",
        "smri_CSF_vit_features.npy",
        "smri_roi_voxel_info.json",
    )
    return all(os.path.isfile(os.path.join(FEATURE_DIR, name)) for name in required)


# Main classification function

def run_classification_experiment(
    selected_channels=None,
    combination_name=None,
    random_state=default_random_state,
    cv_folds=default_cv_folds,
    data_dir=None,
    force_recompute_features=False,
):
    # Load split and multimodal features -> concatenate -> inner K-fold MLP; fMRI reads cache only
    if selected_channels is None:
        selected_channels = list(default_channels)
    if combination_name is None:
        combination_name = default_combination_name

    print("=" * 80)
    print("MUSA Classification Experiment")
    print("=" * 80)
    print("MUSA directory: %s" % (PROJECT_ROOT,))
    print("Combination name: %s" % (combination_name,))
    print("Channels: %s" % (" + ".join(selected_channels),))
    print("Random seed: %d" % (random_state,))
    print("CV folds: %d" % (cv_folds,))
    print("Start time: %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))

    # 0. Set random seed
    set_global_seed(random_state)
    
    # 1. Load data split
    if data_dir is None:
        data_dir = SCRIPT_DIR
    
    print("\nStep 1: Loading data split...")
    train_indices, test_indices, labels, file_list = load_saved_data_split_only(
        random_state, holdout_test_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device: %s" % (device,))
    
    # 2.1 Load sMRI features
    print("\nStep 2: Loading sMRI ViT features...")
    if not smri_vit_features_cache_ready(force_recompute_features):
        raise FileNotFoundError(
            "Complete sMRI ViT feature cache not found (requires "
            "smri_GM_vit_features.npy, smri_WM_vit_features.npy, "
            "smri_CSF_vit_features.npy, and smri_roi_voxel_info.json). "
            "Run code/extract_roi_vit_features.py first to extract imaging voxel features."
        )
    smri_channels_dict, _ = extract_smri_all_channels_vit_features(
        script_dir=data_dir,
        file_list=file_list,
        device=device,
        script_dir_for_save=PROJECT_ROOT,
        force_recompute=force_recompute_features,
    )
    print("  ✓ sMRI channels: %s" % (list(smri_channels_dict.keys())))
    for ch_name in selected_channels:
        if ch_name in smri_channels_dict:
            print("  %s feature shape: %s" % (ch_name, smri_channels_dict[ch_name].shape))
    
    # 2.2 Load FC matrices
    print("\nStep 3: Loading FC matrices...")
    fc_matrices = load_fc_matrices(data_dir, file_list)
    print("  ✓ FC matrix shape: %s" % (fc_matrices.shape,))
    
    print("\nStep 4: Loading fMRI ViT features...")
    if not fmri_vit_features_cache_ready(PROJECT_ROOT, False):
        raise FileNotFoundError(
            "Complete fMRI ViT feature cache not found. "
            "Run code/extract_roi_vit_features.py first to extract imaging voxel features."
        )
    fmri_channels_dict = extract_fmri_all_channels_vit_features(
        fmri_roi_voxels=None,
        roi_voxel_counts=None,
        roi_ids=None,
        device=device,
        script_dir=PROJECT_ROOT,
        force_recompute=False,
    )
    print("  ✓ Loaded saved fMRI ViT features")
    
    for ch_name in selected_channels:
        if ch_name in fmri_channels_dict:
            print("  %s feature shape: %s" % (ch_name, fmri_channels_dict[ch_name].shape))
    
    # 3. Concatenate features
    print("\nStep 5: Concatenating features...")
    concatenated = concatenate_selected_channels(
        smri_channels_dict,
        fc_matrices,
        fmri_channels_dict,
        selected_channels,
        include_fc=True,
    )
    print("  Concatenated feature shape: %s" % (concatenated.shape,))  # [N, 116, per_roi_dim]
    
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

        sensitivity = metrics.get("sensitivity")
        specificity = metrics.get("specificity")
        precision = metrics.get("precision")
        confusion_matrix = metrics.get("confusion_matrix")

        print(
            "  ✓ Fold %d: ACC=%.4f, SEN=%.4f, SPE=%.4f, PRE=%.4f, F1=%.4f, AUC=%.4f"
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
        
        print("ACC: %.4f ± %.4f (%.2f%% ± %.2f%%)" % (acc_mean,
            acc_std,
            acc_mean * 100.0,
            acc_std * 100.0))
        if sen_mean is not None:
            print("SEN: %.4f ± %.4f" % (sen_mean, sen_std))
        if spe_mean is not None:
            print("SPE: %.4f ± %.4f" % (spe_mean, spe_std))
        if pre_mean is not None:
            print("PRE: %.4f ± %.4f" % (pre_mean, pre_std))
        if f1_mean is not None:
            print("F1 : %.4f ± %.4f" % (f1_mean, f1_std))
        if auc_mean is not None:
            print("AUC: %.4f ± %.4f" % (auc_mean, auc_std))
        
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
        f"classification_{combination_name}_seed{random_state}_holdout80_inner{cv_folds}fold.json",
    )
    
    output_data = {
        "experiment_name": f"Classification: {' + '.join(selected_channels)}",
        "combination_name": combination_name,
        "channels": selected_channels,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "musa_dir": PROJECT_ROOT,
        "fold_results": fold_results,
        "summary_metrics": summary_metrics,
    }
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n✓ Experiment results saved to: %s" % (result_file))
    print("✓ Experiment completed, end time: %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
    
    return output_data


if __name__ == '__main__':
    main()
