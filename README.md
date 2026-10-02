# MUSA

MUSA is a multimodal neuroimaging pipeline for binary classification of MDD vs. HC, combining:

- sMRI ROI-level 3D ViT features (`GM`, `WM`, `CSF`)
- fMRI ROI-level 3D ViT features (`ReHo`, `DC`, `ALFF`, `VMHC`)
- Functional connectivity (FC) matrices

The project covers the full workflow from data splitting and ROI feature extraction to single-combination classification evaluation, using inner cross-validation on the training set and a held-out test set.

## Key Features

- Fixed random seed for reproducible data splits
- ROI extraction based on the AAL116 atlas
- ROI-level 3D ViT encoding
- Multimodal feature concatenation (e.g. `WM+FC+ReHo+DC`)
- 5-fold inner cross-validation on the training set + independent test-set evaluation
- Results exported automatically as JSON for review and analysis

## Project Structure

```text
MUSA/
├─ code/
│  ├─ data_split_and_paths.py           # step 1: stratified split and cache
│  ├─ extract_roi_vit_features.py       # step 2: ROI voxel + ViT feature extraction
│  ├─ classify_single_combo.py          # step 3: single combination classification
│  └─ network/
│     └─ mua_common.py                  # shared models/utilities (ViT, FC loader, MLP)
├─ data/
│  ├─ label.csv
│  ├─ AAL116.nii
│  ├─ sMRI/
│  │  ├─ GM/ | WM/ | CSF/ 
│  ├─ fMRI/
│  │  ├─ ALFF/
│  │  ├─ DC/
│  │  ├─ ReHo/
│  │  └─ VMHC/
│  └─ FC_matrices/
├─ 3dvit_features/                      # extracted feature caches
├─ results/                             # experiment outputs 
└─ requirements.txt
```

## Requirements

- Python `3.10+` (recommended: `3.10` or `3.11`)
- Linux/macOS (Windows paths may need manual adjustment)
- NVIDIA GPU recommended for faster ROI feature extraction; CPU is supported but slower

### Recommended: virtual environment

`venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Or `conda`:

```bash
conda create -n musa python=3.10 -y
conda activate musa
python -m pip install --upgrade pip
```

### Install dependencies (GPU, CUDA 12.1)

```bash
pip install -r requirements.txt
```

> `requirements.txt` currently pins `torch==2.2.1+cu121` for CUDA 12.1.  
> If your CUDA version differs, install a matching PyTorch build first, then install the remaining packages.

Example (install PyTorch first, then other packages):

```bash
# Example: CUDA 12.1
pip install torch==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# Then install remaining dependencies (avoid reinstalling torch)
pip install numpy==1.26.4 pandas==2.3.3 scipy==1.17.0 scikit-learn==1.8.0 nibabel==5.3.3 tqdm==4.67.1
```

### CPU-only install (no NVIDIA GPU)

```bash
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4 pandas==2.3.3 scipy==1.17.0 scikit-learn==1.8.0 nibabel==5.3.3 tqdm==4.67.1
```

### Quick sanity check after install

```bash
python -c "import numpy,pandas,scipy,sklearn,nibabel,tqdm; print('Dependencies imported successfully')"
```

## Data Layout

### 1) Label file

`data/label.csv` must contain:

- `ID`: sample ID (must match image filename stem)
- `label`: class (`0` for HC, `1` for MDD)

Example:

```csv
ID,label
S1-1-0001,1
S1-1-0002,1
```

### 2) sMRI data

- Expected under `data/sMRI/`
- Preferred channel directories: `GM`, `WM`, `CSF`
- Filename stem must match `label.csv` `ID`, e.g.:
  - `data/sMRI/WM/S1-1-0001.nii.gz`

### 3) fMRI data

Expected directories and naming patterns:

- `data/fMRI/ALFF/ALFFMap_{ID}.nii.gz`
- `data/fMRI/DC/DegreeCentrality_PositiveWeightedSumBrainMap_{ID}.nii.gz`
- `data/fMRI/ReHo/ReHoMap_{ID}.nii.gz`
- `data/fMRI/VMHC/zVMHCMap_{ID}.nii.gz` (or `VMHCMap_{ID}.nii.gz`)

### 4) FC matrices

`data/FC_matrices/` must contain one `.mat` file per sample:

- Filename: `{ID}.mat`
- Matrix key in `.mat`: `FC`
- Matrix shape: `(116, 116)`

### 5) Brain atlas

- `data/AAL116.nii` is required for ROI extraction.

## Quick Start

Run from the repository root (`MUSA`).

### Step 1: Generate train/test split cache

```bash
python code/data_split_and_paths.py
```

Outputs:

- `data/data_split.npy`
- `data/labels.npy`
- `data/file_list.npy`

### Step 2: Extract ROI ViT features

```bash
python code/extract_roi_vit_features.py
```

Outputs (under `3dvit_features/`):

- `smri_GM_vit_features.npy`
- `smri_WM_vit_features.npy`
- `smri_CSF_vit_features.npy`
- `fmri_ReHo_vit_features.npy`
- `fmri_DC_vit_features.npy`
- `fmri_ALFF_vit_features.npy`
- `fmri_VMHC_vit_features.npy`
- `smri_roi_voxel_info.json`
- `fmri_roi_voxel_info.json`

### Step 3: Run single-combination classification

```bash
python code/classify_single_combo.py
```

Default experiment settings:

- Combination: `WM+FC+ReHo+DC`
- Seed: `42`
- Holdout test size: `0.2`
- Inner CV folds: `5`

Results are saved to `results/`, e.g.:

- `classification_WM+FC+ReHo+DC_seed42_holdout80_inner5fold.json`

## Quick Run with Pre-extracted Features (Skip ROI ViT Extraction)

If `3dvit_features/` already contains **all 9 feature cache files**, you can skip the time-consuming Step 2 (ROI ViT extraction) and run classification directly. This path is suitable when you have no GPU or want to reproduce classification results quickly.

### Download pre-extracted features

Pre-extracted ViT features are available on Google Drive. Download and place them under `3dvit_features/` at the project root:

**[3dvit_features - Google Drive](https://drive.google.com/drive/folders/16lA0-9cZI0YnA7fe0Gz6856OewzHc9zi?usp=sharing)**

Steps:

1. Open the link above and download all files in the folder (or download individually).
2. Place the files in `MUSA/3dvit_features/` (same level as `code/` and `data/`).
3. Verify that all 9 files listed below are present.

### Required files

`3dvit_features/` must contain the following (all required):

| File | Description |
|------|-------------|
| `smri_GM_vit_features.npy` | sMRI GM channel ViT features |
| `smri_WM_vit_features.npy` | sMRI WM channel ViT features |
| `smri_CSF_vit_features.npy` | sMRI CSF channel ViT features |
| `fmri_ReHo_vit_features.npy` | fMRI ReHo channel ViT features |
| `fmri_DC_vit_features.npy` | fMRI DC channel ViT features |
| `fmri_ALFF_vit_features.npy` | fMRI ALFF channel ViT features |
| `fmri_VMHC_vit_features.npy` | fMRI VMHC channel ViT features |
| `smri_roi_voxel_info.json` | sMRI ROI voxel metadata |
| `fmri_roi_voxel_info.json` | fMRI ROI voxel metadata |

### Workflow when features are complete

You still need labels, FC matrices, and the data-split cache under `data/`. **Raw sMRI/fMRI images are not required for classification**, but sMRI file paths are still needed when generating the split so they can be aligned with `label.csv`.

```bash
# 1. Install dependencies (see Requirements above)
pip install -r requirements.txt

# 2. Generate train/test split (must match the sample set used for pre-extracted features)
python code/data_split_and_paths.py

# 3. Skip feature extraction — this command auto-skips when cache is complete
python code/extract_roi_vit_features.py
# Look for "Feature cache is complete; skipping ..." in the terminal

# 4. Run classification
python code/classify_single_combo.py
```

**Notes:**

- Pre-extracted features are tied to a specific sample set and the order in `file_list.npy`. Use the **same** `data/label.csv` and sMRI naming as when the features were generated.
- Classification still requires FC matrices for every sample under `data/FC_matrices/` (`.mat` files, key `FC`, shape `(116, 116)`).
- Step 3 classification is MLP-based and runs on CPU; a GPU speeds up training.
- If a `.npy` file is missing or corrupted, set `default_force_recompute_features = True` in `code/extract_roi_vit_features.py` and re-run Step 2 to force recomputation.

## Reproducibility

- Main scripts fix the random seed (default `set_global_seed(42)`).
- If feature cache already exists, the extraction script skips recomputation. To force re-extraction, set:
  - `default_force_recompute_features = True`

## Configuration

Most parameters are set as script constants (not CLI flags). Edit as needed:

- `code/classify_single_combo.py`
  - `default_channels`
  - `default_combination_name`
  - `default_random_state`
  - `default_cv_folds`
- `code/extract_roi_vit_features.py`
  - `default_force_recompute_features`

## FAQ

- `FileNotFoundError: AAL116 atlas file not found`  
  Ensure `data/AAL116.nii` exists (only needed when extracting features yourself).

- `FC matrices directory not found` or FC shape/key mismatch  
  Check `data/FC_matrices/`; each file must contain key `FC` with shape `(116, 116)`.

- sMRI/fMRI file not found for a sample ID  
  Ensure image filename stems exactly match `ID` in `label.csv`.

- GPU memory pressure during extraction  
  Use a GPU with more VRAM, or reduce batch-size parameters in the extraction code.
