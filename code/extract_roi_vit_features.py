#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MUSA: ROI voxel and sMRI / fMRI ViT feature extraction

import os
import sys

# Add code directory to path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import json

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from scipy.ndimage import affine_transform
from tqdm import tqdm

from data_split_and_paths import SCRIPT_DIR, FEATURE_DIR, load_file_list_npy
from network.mua_common import extract_roi_vit_features_from_voxels, set_global_seed

# Default tunable parameters
default_force_recompute_features    = False

# Skip voxel and feature extraction if all files below exist and force_recompute is False
_FULL_FEATURE_CACHE_FILENAMES = (
    'fmri_ALFF_vit_features.npy',
    'fmri_DC_vit_features.npy',
    'fmri_ReHo_vit_features.npy',
    'fmri_roi_voxel_info.json',
    'fmri_VMHC_vit_features.npy',
    'smri_GM_vit_features.npy',
    'smri_WM_vit_features.npy',
    'smri_CSF_vit_features.npy',
    'smri_roi_voxel_info.json',
)


def main():
    # Match data split and classification scripts; fix ViT randomness for reproducible npy
    set_global_seed(42)

    # file_list.npy -> task 1 (fMRI all-channel ViT) -> task 2 (sMRI all-channel ViT)
    print("=" * 70)
    print("MUSA ROI and sMRI/fMRI Feature Extraction")
    print("=" * 70)

    print("\nStep 0: Loading file_list.npy...")
    file_list = load_file_list_npy()
    print(f"  Total samples: {len(file_list)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    if not default_force_recompute_features:
        feature_dir = FEATURE_DIR
        missing = [
            fn for fn in _FULL_FEATURE_CACHE_FILENAMES
            if not os.path.exists(os.path.join(feature_dir, fn))
        ]
        if not missing:
            print("\n" + "=" * 70)
            print("Feature cache is complete; skipping fMRI and sMRI ViT extraction")
            print("=" * 70)
            for fn in _FULL_FEATURE_CACHE_FILENAMES:
                print("  %s" % (os.path.join(feature_dir, fn)))
            print("=" * 70)
            print("Set default_force_recompute_features to True to force recomputation")
            return

    print("\n" + "=" * 70)
    print("Task 1: Extract fMRI all-channel ViT features")
    print("=" * 70)
    fmri_roi_voxels, roi_voxel_counts, roi_ids = extract_fmri_roi_voxels(
        script_dir=SCRIPT_DIR,
        file_list=file_list,
        force_recompute=default_force_recompute_features,
    )
    print(f"  ✓ fMRI ROI voxel extraction complete")
    print(f"  Channels: {len(fmri_roi_voxels)}")
    print(f"  ROIs: {len(roi_ids)}")

    fmri_channels_dict = extract_fmri_all_channels_vit_features(
        fmri_roi_voxels=fmri_roi_voxels,
        roi_voxel_counts=roi_voxel_counts,
        roi_ids=roi_ids,
        device=device,
        script_dir=SCRIPT_DIR,
        force_recompute=default_force_recompute_features,
    )
    print(f"  ✓ fMRI all-channel ViT feature extraction complete")
    print(f"  Channels: {len(fmri_channels_dict)}")
    for ch_name, features in fmri_channels_dict.items():
        print(f"    {ch_name}: {features.shape}")

    print("\n" + "=" * 70)
    print("Task 2: Extract sMRI all-channel ViT features")
    print("=" * 70)
    smri_channels_dict, roi_ids_smri = extract_smri_all_channels_vit_features(
        script_dir=SCRIPT_DIR,
        file_list=file_list,
        device=device,
        script_dir_for_save=SCRIPT_DIR,
        force_recompute=default_force_recompute_features,
    )
    print(f"  ✓ sMRI all-channel ViT feature extraction complete")
    print(f"  Channels: {len(smri_channels_dict)}")
    for ch_name, features in smri_channels_dict.items():
        print(f"    {ch_name}: {features.shape}")

    print("\n" + "=" * 70)
    print("Feature extraction pipeline complete")
    print("=" * 70)
    print("  Task 1 (fMRI all-channel ViT): ✓ success")
    print("  Task 2 (sMRI all-channel ViT): ✓ success")
    print("=" * 70)


def _project_root_from_path(path):
    # Extract MUSA project root from absolute path
    if not path:
        return None
    parts = path.replace("\\", "/").split("/")
    marker = "MUSA"
    if marker in parts:
        idx = parts.index(marker)
        return os.sep.join(parts[: idx + 1])
    return None


def _sample_ids_from_file_list(file_list):
    # Sample ID list aligned with NIfTI filename stems
    sample_ids = []
    for fpath in file_list:
        fname = os.path.basename(fpath)
        if fname.endswith('.nii.gz'):
            stem = fname[:-7]
        elif fname.endswith('.nii'):
            stem = fname[:-4]
        else:
            stem, _ = os.path.splitext(fname)
        sample_ids.append(stem)
    return sample_ids


def _write_smri_roi_voxel_info_json(feature_dir, roi_ids, roi_voxel_counts, file_list):
    # sMRI per-ROI voxel counts + sample IDs; same structure as fmri_roi_voxel_info.json
    os.makedirs(feature_dir, exist_ok=True)
    roi_list = roi_ids.tolist() if hasattr(roi_ids, "tolist") else list(roi_ids)
    roi_list = [int(x) for x in roi_list]
    counts = {int(k): int(v) for k, v in roi_voxel_counts.items()}
    base_note = (
        "sMRI: per-ROI voxel counts on unified grid (121,145,121); "
        "voxel counts are consistent across samples after preprocessing"
    )
    roi_info = {
        "roi_ids": roi_list,
        "roi_voxel_counts": counts,
        "sample_ids": _sample_ids_from_file_list(file_list),
        "note": base_note,
    }
    path = os.path.join(feature_dir, "smri_roi_voxel_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(roi_info, f, ensure_ascii=False, indent=2)
    print(f"  ✓ sMRI ROI voxel info saved: {path}")


def fmri_vit_features_cache_ready(script_dir, force_recompute):
    # Whether all four fMRI channel npy files exist (same check as in extract_fmri)
    if not script_dir or force_recompute:
        return False
    possible_feature_dirs = [FEATURE_DIR]
    for ch_name in ('ReHo', 'DC', 'ALFF', 'VMHC'):
        ok = False
        for feat_dir in possible_feature_dirs:
            if not os.path.exists(feat_dir):
                continue
            p = os.path.join(feat_dir, f'fmri_{ch_name}_vit_features.npy')
            if os.path.exists(p):
                ok = True
                break
        if not ok:
            return False
    return True


def load_aal116_atlas(script_dir):
    # Load AAL116.nii atlas
    atlas_file = os.path.join(script_dir, 'AAL116.nii')
    if not os.path.exists(atlas_file):
        raise FileNotFoundError(f"AAL116 atlas file not found: {atlas_file}")
    
    atlas_img = nib.load(atlas_file)
    return atlas_img


def resample_atlas_to_image(atlas_img, target_img, interpolation='nearest'):
    # Resample AAL atlas affine to target grid; interpolation controls nearest/linear
    # A) Unify canonical orientation
    # Apply as_closest_canonical to atlas and img before affine computation
    # to ensure consistent orientation
    atlas_img = nib.as_closest_canonical(atlas_img)
    target_img = nib.as_closest_canonical(target_img)
    
    # Target image info
    target_shape = target_img.shape[:3]
    target_affine = target_img.affine
    
    # Atlas info
    atlas_data = atlas_img.get_fdata()
    atlas_affine = atlas_img.affine
    
    # Transform from target voxel coords to atlas voxel coords
    # Correct transform: atlas_voxel = inv(atlas_affine) @ target_affine @ target_voxel
    # i.e. T = inv(atlas_affine) @ target_affine maps target voxel -> atlas voxel
    # affine_transform needs: output (target) voxel -> input (atlas) voxel
    T = np.linalg.inv(atlas_affine) @ target_affine
    
    # affine_transform parameters:
    # - matrix: 3x3 rotation+scale (target voxel -> atlas voxel)
    # - offset: translation (target voxel -> atlas voxel)
    matrix = T[:3, :3]
    offset = T[:3, 3]
    
    # Set interpolation order
    if interpolation == 'nearest':
        order = 0
    else:
        order = 1
    
    # Affine resampling
    # mode='constant', cval=0: treat out-of-bounds as background (0) to avoid spurious ROI labels at edges
    resampled_data = affine_transform(
        atlas_data,
        matrix,
        offset=offset,
        output_shape=target_shape,
        order=order,
        mode='constant',
        cval=0,
        prefilter=False
    )
    
    # New nibabel image
    resampled_img = nib.Nifti1Image(resampled_data.astype(np.int32), target_affine, target_img.header)
    return resampled_img


def build_unified_smri_grid(file_list, target_shape=(121, 145, 121)):
    # Unified sMRI grid: fixed shape, affine/header from first sample
    if file_list is None or len(file_list) == 0:
        raise ValueError("file_list is empty; cannot build unified sMRI grid")
    ref_img = nib.as_closest_canonical(nib.load(file_list[0]))
    if tuple(ref_img.shape[:3]) != tuple(target_shape):
        print(
            f"First reference sMRI shape {ref_img.shape[:3]} differs from target grid {target_shape}; "
            "will resample to target grid"
        )
    return target_shape, ref_img.affine, ref_img.header


def resample_image_to_grid(source_img, target_shape, target_affine, interpolation='linear'):
    # Resample single image to given shape + affine
    source_img = nib.as_closest_canonical(source_img)
    source_data = source_img.get_fdata().astype(np.float32)
    source_affine = source_img.affine

    # source_voxel = inv(source_affine) @ target_affine @ target_voxel
    T = np.linalg.inv(source_affine) @ target_affine
    matrix = T[:3, :3]
    offset = T[:3, 3]
    order = 1 if interpolation == 'linear' else 0

    resampled_data = affine_transform(
        source_data,
        matrix,
        offset=offset,
        output_shape=target_shape,
        order=order,
        mode='constant',
        cval=0,
        prefilter=(order > 0)
    )
    return resampled_data.astype(np.float32)


# fMRI ROI voxel extraction

def extract_fmri_roi_voxels(script_dir, file_list, force_recompute=False):
    # Four-channel fMRI ROI voxels; read disk per sample
    print("\nExtracting fMRI ROI voxel data...")
    
    # fMRI channel directories
    fmri_channels = {
        'ALFF': os.path.join(script_dir, 'fMRI', 'ALFF'),
        'DC': os.path.join(script_dir, 'fMRI', 'DC'),
        'ReHo': os.path.join(script_dir, 'fMRI', 'ReHo'),
        'VMHC': os.path.join(script_dir, 'fMRI', 'VMHC')
    }
    
    # Check directories exist
    for ch_name, ch_dir in fmri_channels.items():
        if not os.path.exists(ch_dir):
            print(f"fMRI channel directory not found: {ch_dir}")
    
    # Load AAL atlas (nibabel image with affine)
    atlas_img = load_aal116_atlas(script_dir)
    
    # ROI IDs from original template
    atlas_data = atlas_img.get_fdata()
    unique_roi_ids = np.unique(atlas_data)
    unique_roi_ids = unique_roi_ids[unique_roi_ids > 0]
    unique_roi_ids = np.sort(unique_roi_ids)[:116]  # Ensure exactly 116 ROIs
    n_rois = len(unique_roi_ids)
    
    print(f"  Found {n_rois} ROIs")
    print(f"  Total samples: {len(file_list)}")
    print("  Affine resampling: resample AAL atlas to each image grid (nearest interpolation)")
    
    # Store voxel data for all channels
    fmri_roi_voxels = {}
    reference_voxel_counts = None
    
    # Process each channel
    for ch_idx, (ch_name, ch_dir) in enumerate(fmri_channels.items()):
        print(f"  Processing channel {ch_name}...")
        channel_roi_voxels = {roi_id: [] for roi_id in unique_roi_ids}
        
        for file_path in tqdm(file_list, desc=f"    Processing {ch_name}"):
                # Extract stem from sMRI file path
                filename = os.path.basename(file_path)
                if filename.endswith('.nii.gz'):
                    stem = filename[:-7]
                elif filename.endswith('.nii'):
                    stem = filename[:-4]
                else:
                    stem, _ = os.path.splitext(filename)
                
                # Build fMRI file path
                channel_patterns = {
                    'ALFF': [f"ALFFMap_{stem}.nii.gz"],
                    'DC': [f"DegreeCentrality_PositiveWeightedSumBrainMap_{stem}.nii.gz"],
                    'ReHo': [f"ReHoMap_{stem}.nii.gz"],
                    'VMHC': [f"zVMHCMap_{stem}.nii.gz", f"VMHCMap_{stem}.nii.gz"],
                }
                fmri_file = None
                if ch_name in channel_patterns:
                    for pattern in channel_patterns[ch_name]:
                        candidate_file = os.path.join(ch_dir, pattern)
                        if os.path.exists(candidate_file):
                            fmri_file = candidate_file
                            break
                if fmri_file is None:
                    tried = [
                        os.path.join(ch_dir, p)
                        for p in channel_patterns.get(ch_name, [])
                    ]
                    raise FileNotFoundError(
                        f"Channel {ch_name} sample stem={stem!r}: no fMRI file with expected naming; tried: {tried}"
                    )

                img = nib.load(fmri_file)
                img_data = img.get_fdata().astype(np.float32)

                atlas_to_img = resample_atlas_to_image(
                    atlas_img,
                    img,
                    interpolation='nearest'
                )
                atlas_data_resampled = atlas_to_img.get_fdata().astype(np.int32)

                if img_data.std() > 1e-8:
                    img_data = (img_data - img_data.mean()) / (img_data.std() + 1e-8)

                img_data_flat = img_data.flatten()
                atlas_flat = atlas_data_resampled.flatten()

                for roi_id in unique_roi_ids:
                    roi_indices = np.where(atlas_flat == roi_id)[0]
                    roi_voxels = img_data_flat[roi_indices]
                    channel_roi_voxels[roi_id].append(roi_voxels)

        # ========================================================================
        # Per-channel voxel count handling: VMHC vs non-VMHC
        # ========================================================================
        if ch_name != 'VMHC':
            # --------------------------------------------------------------------
            # Non-VMHC channels (ALFF/DC/ReHo): voxel counts identical across samples
            # --------------------------------------------------------------------
            # These channels use standard MNI normalization with the same resampling;
            # ROI voxel counts are identical across samples -> numpy arrays for batch processing
            # --------------------------------------------------------------------
            for roi_id in unique_roi_ids:
                channel_roi_voxels[roi_id] = np.array(channel_roi_voxels[roi_id])
            
            # First non-VMHC channel: record reference voxel counts (for VMHC alignment)
            if reference_voxel_counts is None:
                reference_voxel_counts = {}
                for roi_id in unique_roi_ids:
                    reference_voxel_counts[roi_id] = channel_roi_voxels[roi_id].shape[-1]
                print(f"    Recorded reference voxel counts (from {ch_name} channel)")
        else:
            # VMHC: align to reference voxel counts
            if reference_voxel_counts is None:
                raise ValueError("Reference voxel counts not found; ensure ALFF/DC/ReHo are processed before VMHC")
            
            print(f"    Aligning VMHC to reference voxel counts...")
            aligned_count = 0
            padded_count = 0
            truncated_count = 0
            
            for roi_id in unique_roi_ids:
                target_length = reference_voxel_counts[roi_id]  # Reference voxel count
                aligned_roi_voxels = []
                
                for roi_voxels in channel_roi_voxels[roi_id]:
                    current_length = len(roi_voxels)

                    if current_length == target_length:
                        # Same length; no change
                        aligned_roi_voxels.append(roi_voxels)
                        aligned_count += 1
                    elif current_length < target_length:
                        # Fewer voxels than reference: zero-pad to target length
                        padding_length = target_length - current_length
                        padded_voxels = np.pad(roi_voxels, (0, padding_length), mode='constant', constant_values=0)
                        aligned_roi_voxels.append(padded_voxels)
                        padded_count += 1
                    else:
                        # More voxels than reference: truncate to target length
                        # Keep first N voxels; boundary voxels usually have smaller impact on ROI features
                        truncated_voxels = roi_voxels[:target_length]
                        aligned_roi_voxels.append(truncated_voxels)
                        truncated_count += 1
                
                # Convert to numpy array (all samples now have consistent voxel counts)
                channel_roi_voxels[roi_id] = np.array(aligned_roi_voxels)
            
            print(f"      VMHC alignment stats: match={aligned_count}, pad={padded_count}, truncate={truncated_count}")
        
        fmri_roi_voxels[ch_name] = channel_roi_voxels
        print(f"    ✓ {ch_name} voxel extraction complete")
    
    print(f"  ✓ fMRI ROI voxel extraction complete")
    
    # Per-ROI voxel counts (fMRI; consistent across samples, use first sample)
    # Note: sMRI and fMRI ROI voxel counts may differ due to different image grids
    roi_voxel_counts = {}
    for roi_id in unique_roi_ids:
        # Voxel count from first channel (should match across channels)
        first_channel = list(fmri_roi_voxels.values())[0]
        roi_data = first_channel[roi_id]
        roi_voxel_counts[roi_id] = roi_data.shape[-1]
    
    feature_dir = None
    project_root = _project_root_from_path(script_dir)
    if project_root is not None:
        feature_dir = FEATURE_DIR
    if feature_dir is None:
        feature_dir = FEATURE_DIR
    os.makedirs(feature_dir, exist_ok=True)

    sample_ids = _sample_ids_from_file_list(file_list)

    roi_info = {
        "roi_ids": unique_roi_ids.tolist(),
        "roi_voxel_counts": {int(k): int(v) for k, v in roi_voxel_counts.items()},
        "sample_ids": sample_ids,
        "note": "roi_voxel_counts: per-ROI voxel count; consistent across samples after standardized preprocessing"
    }
    info_path = os.path.join(feature_dir, 'fmri_roi_voxel_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(roi_info, f, ensure_ascii=False, indent=2)
    print(f"  ✓ fMRI ROI voxel info saved: {info_path}")

    return fmri_roi_voxels, roi_voxel_counts, unique_roi_ids


# sMRI all-channel ViT feature extraction

def extract_smri_all_channels_vit_features(script_dir, file_list, device=None, script_dir_for_save=None, 
                                          force_recompute=False):
    # GM/WM/CSF three-channel ViT; load npy from cache first, else extract and save
    print("\nExtracting sMRI all-channel ViT features...")
    
    # Try loading saved per-channel features first
    if script_dir_for_save is None:
        script_dir_for_save = script_dir
    
    smri_channels = ['GM', 'WM', 'CSF']
    smri_channels_dict = {}

    possible_feature_dirs = [FEATURE_DIR]

    # Try cache per channel; extract only missing channels later
    if script_dir_for_save and not force_recompute:
        for ch_name in smri_channels:
            for feat_dir in possible_feature_dirs:
                if not os.path.exists(feat_dir):
                    continue
                ch_feature_file = os.path.join(feat_dir, f'smri_{ch_name}_vit_features.npy')
                if os.path.exists(ch_feature_file):
                    smri_channels_dict[ch_name] = np.load(ch_feature_file)
                    print(f"  ✓ Loaded saved sMRI channel {ch_name} features, shape: {smri_channels_dict[ch_name].shape}")
                    break

    channels_to_extract = [c for c in smri_channels if c not in smri_channels_dict]

    if not channels_to_extract:
        feat_dir_for_meta = None
        for feat_dir in possible_feature_dirs:
            if not os.path.isdir(feat_dir):
                continue
            if all(
                os.path.isfile(os.path.join(feat_dir, f"smri_{c}_vit_features.npy"))
                for c in smri_channels
            ):
                feat_dir_for_meta = feat_dir
                break
        roi_ids = None
        if feat_dir_for_meta is not None:
            smri_meta = os.path.join(feat_dir_for_meta, "smri_roi_voxel_info.json")
            legacy_meta = os.path.join(feat_dir_for_meta, "roi_voxel_info.json")
            if os.path.isfile(smri_meta):
                with open(smri_meta, "r", encoding="utf-8") as f:
                    roi_info = json.load(f)
                roi_ids = np.array(roi_info["roi_ids"])
            elif os.path.isfile(legacy_meta):
                with open(legacy_meta, "r", encoding="utf-8") as f:
                    roi_info = json.load(f)
                roi_ids = np.array(roi_info["roi_ids"])
                roi_info["note"] = (roi_info.get("note") or "") + " (migrated to smri_roi_voxel_info.json)"
                with open(smri_meta, "w", encoding="utf-8") as f:
                    json.dump(roi_info, f, ensure_ascii=False, indent=2)
                print(f"  ✓ Generated {smri_meta} (migrated from roi_voxel_info.json)")
            else:
                ref = smri_channels_dict[smri_channels[0]]
                n_roi = int(ref.shape[1])
                roi_ids = np.arange(1, n_roi + 1, dtype=int)
        if roi_ids is None:
            roi_ids = np.array(range(1, 117))
        print(f"  ✓ Loaded all sMRI channel features, {len(smri_channels_dict)} channels")
        return smri_channels_dict, roi_ids

    if len(channels_to_extract) == len(smri_channels):
        print("  Re-extracting all channel features...")
    else:
        print(f"  sMRI channels without cache; extracting only: {', '.join(channels_to_extract)}")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Load AAL atlas (nibabel image with affine)
    atlas_img = load_aal116_atlas(script_dir)
    
    # ROI IDs
    atlas_data = atlas_img.get_fdata()
    unique_roi_ids = np.unique(atlas_data)
    unique_roi_ids = unique_roi_ids[unique_roi_ids > 0]
    unique_roi_ids = np.sort(unique_roi_ids)[:116]  # Ensure exactly 116 ROIs
    roi_ids = np.array(sorted(unique_roi_ids))
    
    print(f"  Found {len(roi_ids)} ROIs")
    print("  Unified grid: resample AAL to fixed sMRI grid (121,145,121), then extract")

    # Unified target grid
    unified_shape, unified_affine, unified_header = build_unified_smri_grid(file_list, target_shape=(121, 145, 121))
    unified_target_img = nib.Nifti1Image(np.zeros(unified_shape, dtype=np.float32), unified_affine, unified_header)
    atlas_unified_img = resample_atlas_to_image(atlas_img, unified_target_img, interpolation='nearest')
    atlas_unified_flat = atlas_unified_img.get_fdata().astype(np.int32).flatten()
    
    last_roi_voxel_counts = None
    # Extract ViT only for sMRI channels missing cache
    for ch_name in channels_to_extract:
        print(f"  Processing channel {ch_name}...")
        channel_dir = os.path.join(script_dir, 'sMRI', ch_name)
        
        if not os.path.exists(channel_dir):
            print(f"  sMRI channel directory not found: {channel_dir}")
            if file_list:
                N = len(file_list)
                smri_channels_dict[ch_name] = np.zeros((N, 116, 768), dtype=np.float32)
            continue
        
        roi_voxel_arrays = {roi_id: [] for roi_id in roi_ids}

        print(f"    Extracting {ch_name} ROI voxel data...")
        for file_path in tqdm(file_list, desc=f"    Processing {ch_name}"):
            filename = os.path.basename(file_path)
            channel_file = os.path.join(channel_dir, filename)

            if not os.path.exists(channel_file):
                raise FileNotFoundError(f"sMRI file not found: {channel_file}")

            img = nib.load(channel_file)
            img_data = resample_image_to_grid(
                img,
                target_shape=unified_shape,
                target_affine=unified_affine,
                interpolation='linear'
            )

            if img_data.std() > 1e-8:
                img_data = (img_data - img_data.mean()) / (img_data.std() + 1e-8)

            img_data_flat = img_data.flatten()
            atlas_flat = atlas_unified_flat

            for roi_id in roi_ids:
                roi_indices = np.where(atlas_flat == roi_id)[0]
                roi_voxels = img_data_flat[roi_indices]
                roi_voxel_arrays[roi_id].append(roi_voxels)

        roi_voxel_counts = {}
        for roi_id in roi_ids:
            if len(roi_voxel_arrays[roi_id]) == 0:
                raise ValueError(f"ROI {roi_id} has no available voxel data")
            roi_voxel_counts[roi_id] = len(roi_voxel_arrays[roi_id][0])

        last_roi_voxel_counts = {int(k): int(v) for k, v in roi_voxel_counts.items()}

        for roi_id in roi_ids:
            roi_voxel_arrays[roi_id] = np.stack(roi_voxel_arrays[roi_id], axis=0).astype(np.float32)

        print(f"    ✓ {ch_name} ROI voxel extraction complete")
        print(f"    Note: after standardized preprocessing, voxel counts are consistent across samples for batch processing")

        print(f"    Extracting {ch_name} ViT features...")
        roi_vit_features = extract_roi_vit_features_from_voxels(
            roi_voxel_arrays,
            roi_voxel_counts,
            roi_ids,
            device=device,
            batch_size=8,
            embed_dim=768,
            use_adaptive_model=True
        )

        N = list(roi_vit_features.values())[0].shape[0]
        features_list = []
        for roi_id in roi_ids:
            features_list.append(roi_vit_features[roi_id])

        smri_channels_dict[ch_name] = np.stack(features_list, axis=1)
        print(f"    ✓ {ch_name} ViT feature shape: {smri_channels_dict[ch_name].shape}")

        if script_dir_for_save:
            possible_feature_dirs = [FEATURE_DIR]

            for feat_dir in possible_feature_dirs:
                os.makedirs(feat_dir, exist_ok=True)
                ch_feature_file = os.path.join(feat_dir, f'smri_{ch_name}_vit_features.npy')
                np.save(ch_feature_file, smri_channels_dict[ch_name])
                print(f"    ✓ Saved channel {ch_name} features: {ch_feature_file}")

    if script_dir_for_save and last_roi_voxel_counts is not None:
        meta_dirs = [FEATURE_DIR]
        seen = set()
        for fd in meta_dirs:
            if fd in seen or not fd:
                continue
            seen.add(fd)
            _write_smri_roi_voxel_info_json(fd, roi_ids, last_roi_voxel_counts, file_list)

    return smri_channels_dict, roi_ids


# fMRI all-channel ViT feature extraction

def extract_fmri_all_channels_vit_features(fmri_roi_voxels, roi_voxel_counts, roi_ids, device=None,
                                           sample_batch_size=None, script_dir=None, force_recompute=False):
    # Four-channel fMRI ViT; when fmri_roi_voxels is None, load per-channel npy from disk only
    print("\nExtracting fMRI all-channel ViT features (per-channel dict, no fusion)...")
    
    # Prefer already saved per-channel features
    if script_dir and not force_recompute:
        possible_feature_dirs = [FEATURE_DIR]
        
        all_channels_exist = True
        fmri_channels_dict = {}
        fmri_channel_names = ['ReHo', 'DC', 'ALFF', 'VMHC']
        for ch_name in fmri_channel_names:
            found = False
            for feat_dir in possible_feature_dirs:
                if not os.path.exists(feat_dir):
                    continue
                ch_feature_file = os.path.join(feat_dir, f'fmri_{ch_name}_vit_features.npy')
                if os.path.exists(ch_feature_file):
                    fmri_channels_dict[ch_name] = np.load(ch_feature_file)
                    print(f"  ✓ Loaded saved fMRI channel {ch_name} features, shape: {fmri_channels_dict[ch_name].shape}")
                    found = True
                    break
            if not found:
                all_channels_exist = False
                break
        
        if all_channels_exist and len(fmri_channels_dict) == 4:
            print(f"  ✓ Loaded all fMRI channel features, {len(fmri_channels_dict)} channels")
            return fmri_channels_dict
        else:
            print(f"  Not all fMRI channel feature files found; re-extracting...")
    
    # Device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    first_channel = list(fmri_roi_voxels.values())[0]
    first_roi = list(first_channel.values())[0]
    total_samples = first_roi.shape[0]
    
    if sample_batch_size is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 40:
                sample_batch_size = 500
            elif gpu_memory_gb >= 20:
                sample_batch_size = 300
            else:
                sample_batch_size = 200
        else:
            sample_batch_size = 100
    
    use_batch_processing = total_samples > sample_batch_size
    if use_batch_processing:
        pass
    
    fmri_channels_dict = {}
    
    for ch_name, channel_roi_voxels in fmri_roi_voxels.items():
        print(f"  Processing channel {ch_name}...")
        
        padded_channel_roi_voxels = {}
        for roi_id in roi_ids:
            padded_channel_roi_voxels[roi_id] = np.array(channel_roi_voxels[roi_id])
        
        if use_batch_processing:
            num_sample_batches = (total_samples + sample_batch_size - 1) // sample_batch_size
            batch_features_list = []
            
            for sample_batch_idx in range(num_sample_batches):
                start_idx = sample_batch_idx * sample_batch_size
                end_idx = min((sample_batch_idx + 1) * sample_batch_size, total_samples)
                
                batch_channel_roi_voxels = {}
                for roi_id in roi_ids:
                    batch_channel_roi_voxels[roi_id] = padded_channel_roi_voxels[roi_id][
                        start_idx:end_idx
                    ]
                
                batch_roi_vit_features = extract_roi_vit_features_from_voxels(
                    batch_channel_roi_voxels,
                    roi_voxel_counts,
                    roi_ids,
                    device=device,
                    batch_size=8,
                    embed_dim=768,
                    use_adaptive_model=True,
                )
                
                batch_features_list.append(batch_roi_vit_features)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            roi_vit_features = {}
            for roi_id in roi_ids:
                roi_feat_list = []
                for batch_feat_dict in batch_features_list:
                    roi_feat_list.append(batch_feat_dict[roi_id])
                roi_vit_features[roi_id] = np.vstack(roi_feat_list)
        else:
            roi_vit_features = extract_roi_vit_features_from_voxels(
                padded_channel_roi_voxels,
                roi_voxel_counts,
                roi_ids,
                device=device,
                batch_size=8,
                embed_dim=768,
                use_adaptive_model=True,
            )
        
        features_list = []
        for roi_id in roi_ids:
            features_list.append(roi_vit_features[roi_id])
        
        fmri_channels_dict[ch_name] = np.stack(features_list, axis=1)
        print(f"    ✓ {ch_name} ViT feature shape: {fmri_channels_dict[ch_name].shape}")
        
        if script_dir:
            possible_feature_dirs = [FEATURE_DIR]
            
            for feat_dir in possible_feature_dirs:
                os.makedirs(feat_dir, exist_ok=True)
                ch_feature_file = os.path.join(feat_dir, f'fmri_{ch_name}_vit_features.npy')
                np.save(ch_feature_file, fmri_channels_dict[ch_name])
                print(f"    ✓ Saved channel {ch_name} features: {ch_feature_file}")
    
    return fmri_channels_dict


if __name__ == '__main__':
    main()
