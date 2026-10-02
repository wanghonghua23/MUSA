#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MUSA: compute functional connectivity (FC) matrices from ROI time series

import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io

# Script directory
ROOT = os.path.dirname(os.path.abspath(__file__))

# AAL116 atlas and ROI label files
AAL_MAP_FILE = os.path.join(ROOT, "aal116MNI.nii")
NODE_INDEX_FILE = os.path.join(ROOT, "aal116NodeIndex.1D")
NODE_NAME_FILE = os.path.join(ROOT, "aal116NodeNames.txt")

# Input: ROI time series; output: FC matrices and label table
ROI_SIGNAL_DIR = os.path.join(ROOT, "ROISignals_FunImgARCWF_116")
FC_OUT_DIR = os.path.join(ROOT, "fc")
FIG_OUT_DIR = ROOT
ROI_LABEL_FILE = os.path.join(ROOT, "roi_labels.csv")

ROI_N = 116
SAVE_HEATMAPS = False  # Whether to save FC heatmaps


def load_roi_labels():
    """Load AAL116 ROI indices and names."""
    with open(NODE_INDEX_FILE, "r", encoding="utf-8") as f:
        roi_indices = [int(x.strip()) for x in f if x.strip()]
    with open(NODE_NAME_FILE, "r", encoding="utf-8") as f:
        roi_names = [x.strip() for x in f if x.strip()]

    if len(roi_indices) != ROI_N:
        raise ValueError(
            f"{NODE_INDEX_FILE} should contain {ROI_N} ROIs, got {len(roi_indices)}"
        )
    if len(roi_names) != ROI_N:
        raise ValueError(
            f"{NODE_NAME_FILE} should contain {ROI_N} ROIs, got {len(roi_names)}"
        )

    return roi_indices, roi_names


def save_roi_label_table(roi_indices, roi_names):
    """Save ROI label lookup table (matrix row/col <-> ROI index/name)."""
    df = pd.DataFrame(
        {
            "matrix_index": np.arange(ROI_N, dtype=int),
            "roi_index": roi_indices,
            "roi_name": roi_names,
        }
    )
    df.to_csv(ROI_LABEL_FILE, index=False)
    print("Saved:", ROI_LABEL_FILE)


def list_mat_files(folder):
    """List all ROISignals_*.mat files in a directory."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"ROI time series directory not found: {folder}")
    files = [f for f in os.listdir(folder) if f.startswith("ROISignals_") and f.endswith(".mat")]
    files.sort()
    if len(files) == 0:
        raise ValueError(f"No ROISignals_*.mat files found in {folder}")
    return [os.path.join(folder, f) for f in files]


def extract_subject_id(mat_path):
    """Extract subject ID from filename (ROISignals_{ID}.mat)."""
    name = os.path.basename(mat_path)
    return name[len("ROISignals_"):-len(".mat")]


def load_roi_signals(mat_path):
    """Load ROI time series for one subject; expected shape [timepoints, 116]."""
    mat_data = scipy.io.loadmat(mat_path)
    if "ROISignals" not in mat_data:
        raise KeyError(f"Variable ROISignals not found in {mat_path}")

    signals = mat_data["ROISignals"]
    if signals.ndim != 2:
        raise ValueError(
            f"{mat_path} ROISignals must be 2D, got shape={signals.shape}"
        )
    if signals.shape[1] != ROI_N:
        raise ValueError(
            f"{mat_path} ROISignals must have exactly {ROI_N} columns, got shape={signals.shape}"
        )

    signals = signals.astype(np.float64)
    if not np.isfinite(signals).all():
        raise ValueError(f"{mat_path} ROISignals contains NaN/Inf")
    return signals


def compute_fc_matrix(signals):
    """Compute Pearson correlation matrix across ROI columns."""
    fc = np.corrcoef(signals, rowvar=False)
    # fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    return fc.astype(np.float32)


def save_fc_heatmap(fc, labels, sid):
    """Save FC heatmap (optional)."""
    os.makedirs(FIG_OUT_DIR, exist_ok=True)
    plt.figure(figsize=(15, 12))
    plt.imshow(fc, interpolation="nearest", cmap="RdBu_r", vmax=0.8, vmin=-0.8)
    plt.colorbar()
    plt.xticks(range(ROI_N), labels, rotation=90, fontsize=6)
    plt.yticks(range(ROI_N), labels, fontsize=6)
    plt.subplots_adjust(left=0.12, bottom=0.25, top=0.95, right=0.88)
    out_png = os.path.join(FIG_OUT_DIR, f"FC_{sid}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    """Main pipeline: validate atlas -> load ROI labels -> compute and save FC per subject."""
    if not os.path.isfile(AAL_MAP_FILE):
        raise FileNotFoundError(f"AAL atlas not found: {AAL_MAP_FILE}")
    nib.load(AAL_MAP_FILE)

    os.makedirs(FC_OUT_DIR, exist_ok=True)
    roi_indices, roi_names = load_roi_labels()
    save_roi_label_table(roi_indices, roi_names)

    mat_files = list_mat_files(ROI_SIGNAL_DIR)
    print("Found ROI time series files:", len(mat_files))
    print("ROI order: column 0 ->", roi_names[0], ", column 115 ->", roi_names[-1])

    for i, mat_path in enumerate(mat_files, start=1):
        sid = extract_subject_id(mat_path)
        signals = load_roi_signals(mat_path)
        fc = compute_fc_matrix(signals)

        out_path = os.path.join(FC_OUT_DIR, f"{sid}.mat")
        scipy.io.savemat(out_path, {"FC": fc})

        if SAVE_HEATMAPS:
            save_fc_heatmap(fc, roi_names, sid)

        if i % 100 == 0 or i == len(mat_files):
            print(f"Processed {i}/{len(mat_files)}")

    print("Done.")


if __name__ == "__main__":
    main()
