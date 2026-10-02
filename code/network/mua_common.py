#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MUSA: shared logic for 3D ViT, ROI voxel features, FC, and MLP classification

import gzip
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import nibabel as nib
from scipy.ndimage import zoom, affine_transform
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Script directory
_NETWORK_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(_NETWORK_DIR)
script_dir = CODE_DIR

def set_global_seed(random_state: int):
    # Set Python/numpy/torch/cuda random seeds for reproducibility
    import random
    
    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        uda = getattr(torch, "use_deterministic_algorithms", None)
        if uda is not None:
            uda(True, warn_only=True)

# Part 2: 3D Vision Transformer model

# 3D patch embedding; img_size is passed dynamically per ROI by create_roi_vit_model
class PatchEmbedding3D(nn.Module):
    def __init__(self, img_size=(121, 145, 121), patch_size=(8, 8, 8), in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        
        self.projection = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        x = self.projection(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention3D(nn.Module):
    # Multi-head self-attention
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock3D(nn.Module):
    # Transformer encoder block (LN + MHA + MLP)
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention3D(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# 3D ViT; img_size is set dynamically per ROI by create_roi_vit_model
class VisionTransformer3D(nn.Module):
    def __init__(
        self,
        img_size=(121, 145, 121),
        patch_size=(8, 8, 8),
        in_channels=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=2
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbedding3D(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock3D(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        cls_token_final = x[:, 0]
        out = self.head(cls_token_final)
        return out
    
    def extract_patch_features(self, x):
        # Return patch token features and spatial grid shape (cls excluded for ROI pooling)
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        patch_features = x[:, 1:]
        
        D, H, W = self.img_size
        pD, pH, pW = self.patch_size
        spatial_shape = (D // pD, H // pH, W // pW)
        
        return patch_features, spatial_shape


def load_aal116_atlas(script_dir):
    # Load AAL116 atlas template
    atlas_file = os.path.join(script_dir, 'AAL116.nii')
    if not os.path.exists(atlas_file):
        atlas_file = os.path.join(os.path.dirname(script_dir), 'AAL116.nii')
        if not os.path.exists(atlas_file):
            raise FileNotFoundError(f"AAL116 atlas template not found: {atlas_file}")
    
    atlas_img = nib.load(atlas_file)
    atlas_data = atlas_img.get_fdata()
    return atlas_data.astype(np.int32), atlas_img


def align_atlas_to_image(atlas, target_shape):
    # Align atlas voxel grid to target_shape via zoom
    if atlas.shape == target_shape:
        return atlas
    
    zoom_factors = (
        target_shape[0] / atlas.shape[0],
        target_shape[1] / atlas.shape[1],
        target_shape[2] / atlas.shape[2]
    )
    
    aligned_atlas = zoom(atlas, zoom_factors, order=0).astype(np.int32)
    assert aligned_atlas.shape == target_shape
    return aligned_atlas


def get_roi_mask_indices(atlas, roi_ids=None):
    # Flat atlas voxel indices per ROI
    if roi_ids is None:
        unique_roi_ids = np.unique(atlas)
        unique_roi_ids = unique_roi_ids[unique_roi_ids > 0]
        unique_roi_ids = np.sort(unique_roi_ids)
    else:
        unique_roi_ids = np.array(roi_ids)
    
    roi_mask_indices = {}
    atlas_flat = atlas.flatten()
    
    for roi_id in unique_roi_ids:
        indices = np.where(atlas_flat == roi_id)[0]
        if len(indices) > 0:
            roi_mask_indices[roi_id] = indices
    
    return roi_mask_indices, unique_roi_ids


# Part 3: ROI voxel feature extraction

class ROIVoxelDataset(Dataset):
    # Single ROI voxel sequence -> pad/truncate to 3D grid for ViT input
    def __init__(self, roi_voxel_data, roi_id, target_3d_shape=None):
        self.roi_voxel_data = roi_voxel_data
        self.roi_id = roi_id
        self.N = roi_voxel_data.shape[0]
        self.X_roi = roi_voxel_data.shape[1]
        self.target_3d_shape = target_3d_shape
        
        if target_3d_shape is None:
            self.target_3d_shape = self._find_optimal_shape(self.X_roi)
    
    def _find_optimal_shape(self, num_voxels):
        # num_voxels -> approximate cubic (D, H, W)
        cube_root = int(np.round(num_voxels ** (1/3)))
        
        for d in range(cube_root, cube_root + 10):
            for h in range(cube_root, cube_root + 10):
                for w in range(cube_root, cube_root + 10):
                    if d * h * w >= num_voxels:
                        return (d, h, w)
        
        return (cube_root + 5, cube_root + 5, cube_root + 5)
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        voxels = self.roi_voxel_data[idx]
        
        D, H, W = self.target_3d_shape
        target_size = D * H * W
        
        if len(voxels) < target_size:
            padded_voxels = np.zeros(target_size, dtype=voxels.dtype)
            padded_voxels[:len(voxels)] = voxels
            voxels = padded_voxels
        elif len(voxels) > target_size:
            voxels = voxels[:target_size]
        
        roi_3d = voxels.reshape(D, H, W)
        roi_3d = (roi_3d - roi_3d.mean()) / (roi_3d.std() + 1e-8)
        roi_3d = torch.from_numpy(roi_3d).float().unsqueeze(0)
        
        return roi_3d


def create_roi_vit_model(roi_3d_shape, patch_size, embed_dim=768, depth=6, num_heads=8):
    # Instantiate ViT per ROI 3D shape (num_classes=0 for patch features)
    model = VisionTransformer3D(
        img_size=roi_3d_shape,
        patch_size=patch_size,
        in_channels=1,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_classes=0
    )
    return model


def extract_roi_vit_features_from_voxels(
    roi_voxel_arrays,
    roi_voxel_counts,
    roi_ids,
    device,
    batch_size=8,
    embed_dim=768,
    use_adaptive_model=True
):
    # Per ROI: adaptive patch/depth, DataLoader inference -> {roi_id: [N, embed_dim]}
    # Note: use_adaptive_model is deprecated; adaptive models are always used.
    # The parameter is kept for backward compatibility but its value is ignored.
    roi_vit_features = {}
    
    for roi_id in tqdm(roi_ids, desc="Processing ROIs"):
        if roi_id not in roi_voxel_arrays:
            continue
        
        roi_voxel_data = roi_voxel_arrays[roi_id]
        X_roi = roi_voxel_counts[roi_id]
        N = roi_voxel_data.shape[0]
        
        dataset = ROIVoxelDataset(roi_voxel_data, roi_id)
        D, H, W = dataset.target_3d_shape

        # Percentiles of ROI voxel counts across all ROIs
        all_voxel_counts = list(roi_voxel_counts.values())
        p25 = np.percentile(all_voxel_counts, 25)
        p75 = np.percentile(all_voxel_counts, 75)
        
        if X_roi < p25:
            # Small ROI: smaller patch, shallower network
            patch_size = (2, 2, 2)
            depth = 4
        elif X_roi <= p75:
            # Medium ROI: moderate patch size and depth
            patch_size = (4, 4, 4)
            depth = 6
        else:
            # Large ROI: larger patch, maintained depth
            patch_size = (6, 6, 6)
            depth = 6
        
        pD, pH, pW = patch_size
        if pD > D or pH > H or pW > W:
            patch_size = (min(pD, D), min(pH, H), min(pW, W))
            pD, pH, pW = patch_size
        
        model = create_roi_vit_model(
            (D, H, W), patch_size, embed_dim=embed_dim, depth=depth
        )
        model = model.to(device)
        model.eval()
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        features_list = []
        with torch.no_grad():
            for batch_roi in dataloader:
                batch_roi = batch_roi.to(device)
                
                # Adaptive model; no interpolate needed
                patch_features, spatial_shape = model.extract_patch_features(batch_roi)
                roi_feat = patch_features.mean(dim=1)
                features_list.append(roi_feat.cpu().numpy())
        
        if len(features_list) > 0:
            roi_vit_features[roi_id] = np.vstack(features_list)
    
    return roi_vit_features


# Part 4: FC matrix loading and classification functions

def load_fc_matrices(script_dir, file_list):
    # Load .mat / .mat.gz from FC_matrices by file_list stem -> [N,116,116]
    fc_dir = os.path.join(script_dir, 'FC_matrices')
    if not os.path.exists(fc_dir):
        raise FileNotFoundError(f"FC matrices directory not found: {fc_dir}")

    # Convention: FC matrix field name in .mat files is fixed to this key
    _FC_MAT_KEY = "FC"
    
    fc_matrices = []
    for file_path in file_list:
        filename = os.path.basename(file_path)
        
        # Extract file stem (strip extension)
        if filename.endswith('.nii.gz'):
            stem = filename[:-7]
        elif filename.endswith('.nii'):
            stem = filename[:-4]
        else:
            stem, _ = os.path.splitext(filename)
        
        fc_file = os.path.join(fc_dir, f"{stem}.mat")
        if not os.path.exists(fc_file):
            fc_file = os.path.join(fc_dir, f"{stem}.mat.gz")
        
        if not os.path.exists(fc_file):
            raise FileNotFoundError(
                f"FC matrix file not found for sample ID: {stem}\n"
                f"Tried: {os.path.join(fc_dir, f'{stem}.mat')} and {os.path.join(fc_dir, f'{stem}.mat.gz')}\n"
                f"Please verify FC matrix files exist."
            )
        
        if fc_file.endswith('.gz'):
            with gzip.open(fc_file, 'rb') as f:
                mat_data = loadmat(f)
        else:
            mat_data = loadmat(fc_file)

        if _FC_MAT_KEY not in mat_data:
            raise ValueError(
                f"FC matrix file missing key={_FC_MAT_KEY!r}, file: {fc_file}\n"
                f"Keys in file: {list(mat_data.keys())}"
            )
        fc_matrix = mat_data[_FC_MAT_KEY]

        if fc_matrix.shape != (116, 116):
            raise ValueError(
                f"FC matrix shape incorrect, expected (116, 116), got {fc_matrix.shape}, file: {fc_file}\n"
                f"Please verify FC matrix dimensions."
            )

        fc_matrices.append(fc_matrix.astype(np.float32))
    
    fc_matrices = np.array(fc_matrices)
    return fc_matrices


class MLPClassifier(nn.Module):
    # Fully connected layers + BN + ReLU + Dropout
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], num_classes=2, dropout=0.3):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def flatten_and_classify_mlp(features, labels, train_indices, test_indices, method_name="", 
                             hidden_dims=[640, 512, 384, 256], dropout=0.25, batch_size=64, 
                             num_epochs=400, learning_rate=0.0009, device=None,
                             val_split=0.15, val_indices=None, early_stop_patience=100, early_stop_min_delta=1e-4,
                             enable_early_stop=True, checkpoint_interval=10, use_multi_checkpoint=False):
    # Flatten multi-ROI features -> StandardScaler -> AdamW MLP training, return test metrics
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = features.shape[0]
    features_flat = features.reshape(N, -1)
    input_dim = features_flat.shape[1]
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_flat[train_indices])
    features_normalized = scaler.transform(features_flat)
    
    if val_indices is not None:
        train_idx = np.array(train_indices)
        val_idx = np.array(val_indices)
    else:
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            np.array(train_indices),
            test_size=val_split,
            stratify=labels[train_indices],
            random_state=42,
            shuffle=True,
        )
    
    X_train = torch.FloatTensor(features_normalized[train_idx]).to(device)
    y_train = torch.LongTensor(labels[train_idx]).to(device)
    X_val = torch.FloatTensor(features_normalized[val_idx]).to(device)
    y_val = torch.LongTensor(labels[val_idx]).to(device)
    X_test = torch.FloatTensor(features_normalized[test_indices]).to(device)
    y_test = labels[test_indices]
    
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=2,
        dropout=dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
    
    warmup_epochs = 15
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    model.train()
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_auc = 0.0
    patience_counter = 0
    best_state = None
    checkpoints = []
    
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_loss_val = val_loss.item()
            val_pred = val_outputs.argmax(dim=1).cpu().numpy()
            val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
            val_acc = accuracy_score(y_val.cpu().numpy(), val_pred)
            val_f1 = f1_score(y_val.cpu().numpy(), val_pred)
            val_auc = roc_auc_score(y_val.cpu().numpy(), val_probs)
        
        improved = val_loss_val < best_val_loss - early_stop_min_delta
        if improved:
            best_val_loss = val_loss_val
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_auc = val_auc
            patience_counter = 0
            best_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss_val,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
            }
        else:
            patience_counter += 1
        
        if use_multi_checkpoint and (epoch + 1) % checkpoint_interval == 0:
            model.eval()
            with torch.no_grad():
                val_outputs_checkpoint = model(X_val)
                val_probs_checkpoint = torch.softmax(val_outputs_checkpoint, dim=1)
                val_pred_checkpoint = val_outputs_checkpoint.argmax(dim=1).cpu().numpy()
                val_prob_checkpoint = val_probs_checkpoint[:, 1].cpu().numpy()
                
                val_acc_checkpoint = accuracy_score(y_val.cpu().numpy(), val_pred_checkpoint)
                val_f1_checkpoint = f1_score(y_val.cpu().numpy(), val_pred_checkpoint)
                val_auc_checkpoint = roc_auc_score(y_val.cpu().numpy(), val_prob_checkpoint)
            
            checkpoint_state = {
                'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'optimizer': {k: v for k, v in optimizer.state_dict().items()},
                'scheduler': {k: v for k, v in scheduler.state_dict().items()},
                'epoch': epoch + 1,
                'val_loss': val_loss_val,
                'val_acc': val_acc_checkpoint,
                'val_f1': val_f1_checkpoint,
                'val_auc': val_auc_checkpoint,
            }
            checkpoints.append(checkpoint_state)
            model.train()
        
        if enable_early_stop and patience_counter >= early_stop_patience:
            break
        
        model.train()
    
    if use_multi_checkpoint and len(checkpoints) > 0:
        best_checkpoint = max(checkpoints, key=lambda x: x['val_acc'])
        model.load_state_dict(best_checkpoint['model'])
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_probs = torch.softmax(test_outputs, dim=1)
            y_pred_test = test_outputs.argmax(dim=1).cpu().numpy()
            y_prob_test = test_probs[:, 1].cpu().numpy()
        
        test_acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        auc = roc_auc_score(y_test, y_prob_test)
        
        cm = confusion_matrix(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred_test, average='binary', zero_division=0)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0
        
        return model, test_acc, f1, auc, {
            'accuracy': float(test_acc),
            'sensitivity': float(recall),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1': float(f1),
            'auc': float(auc) if auc is not None else None,
            'confusion_matrix': cm.tolist() if isinstance(cm, np.ndarray) else cm
        }
    else:
        if best_state is not None:
            model.load_state_dict(best_state['model'])
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_probs = torch.softmax(test_outputs, dim=1)
        y_pred_test = test_outputs.argmax(dim=1).cpu().numpy()
        y_prob_test = test_probs[:, 1].cpu().numpy()
    
    test_acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_prob_test)
    
    cm = confusion_matrix(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred_test, average='binary', zero_division=0)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
    
    return model, test_acc, f1, auc, {
        'accuracy': float(test_acc),
        'sensitivity': float(recall),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1': float(f1),
        'auc': float(auc) if auc is not None else None,
        'confusion_matrix': cm.tolist() if isinstance(cm, np.ndarray) else cm
    }


def flatten_and_classify_fc_only(fc_matrices, labels, train_indices, test_indices, method_name="FC-Only",
                                  device=None, use_mlp=True, use_upper_triangle=True):
    # Flatten FC upper triangle + MLP or GBDT
    if use_upper_triangle:
        n_samples = fc_matrices.shape[0]
        n_rois = fc_matrices.shape[1]
        triu_indices = np.triu_indices(n_rois, k=1)
        fc_features = fc_matrices[:, triu_indices[0], triu_indices[1]]
    else:
        fc_features = fc_matrices.reshape(fc_matrices.shape[0], -1)
    
    X_train = fc_features[train_indices]
    X_test = fc_features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if use_mlp:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        y_test_tensor = torch.LongTensor(y_test).to(device)
        
        val_size = int(0.1 * len(X_train_tensor))
        train_size = len(X_train_tensor) - val_size
        indices = torch.randperm(len(X_train_tensor))
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]
        
        X_train_split = X_train_tensor[train_idx]
        y_train_split = y_train_tensor[train_idx]
        X_val = X_train_tensor[val_idx]
        y_val = y_train_tensor[val_idx]
        
        input_dim = X_train.shape[1]
        model = MLPClassifier(input_dim=input_dim, hidden_dims=[512, 256, 128], num_classes=2).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5
        epochs = 100
        batch_size = 32
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_train_split), batch_size):
                batch_X = X_train_split[i:i+batch_size]
                batch_y = y_train_split[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                _, val_preds = torch.max(val_outputs, 1)
                val_acc = (val_preds == y_val).float().mean().item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            scheduler.step()
        
        model.load_state_dict(best_model_state)
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
            _, test_preds = torch.max(test_outputs, 1)
            y_pred = test_preds.cpu().numpy()
    else:
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, test_probs)
    
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
    
    return model, accuracy, f1, auc, {
        'accuracy': float(accuracy),
        'sensitivity': float(recall),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1': float(f1),
        'auc': float(auc) if auc is not None else None,
        'confusion_matrix': cm.tolist() if isinstance(cm, np.ndarray) else cm
    }


if __name__ == '__main__':
    print("mua_common.py is a shared module imported by other scripts.")
