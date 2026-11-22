# MUSA 分类版

用已提取的特征训练原有 MLP，输出 MDD（1）与健康对照 HC（0）的分类结果。实验使用已有的独立测试集划分，并在训练集内做五折交叉验证。每折使用该折的训练子集拟合标准化器和训练 MLP，按验证集表现选择模型，再在同一个独立测试集上评估。

## 安装

建议使用 Python 3.10 或 3.11。在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

程序优先使用可用的 CUDA，也支持 CPU。默认的全量特征训练需要充足内存。

## 输入一个已融合的特征文件

```bash
python code/classify_single_combo.py --features path/to/final_features.npy
```

- `final_features.npy`：数值数组，形状为 `[N, D]`（已展平）或 `[N, 116, D]`（按 ROI 保存）。复现原默认组合时，请按 `WM → FC → ReHo → DC` 的顺序在最后一维拼接，再保存或按行展平。
- `data/labels.npy`：一维标签数组，`0=HC`，`1=MDD`。
- `data/data_split.npy`：原实验缓存，包含 `train_indices`、`test_indices` 和 `random_state`。程序沿用其中的索引和随机种子。此缓存使用 Python pickle，请使用你信任的原实验文件。
- `data/file_list.npy`：可选的原始样本路径列表，用于在预测 CSV 中显示样本 ID。省略时输出特征行索引。请使用可信文件。

特征、标签及样本列表必须沿用同一行序。输入已融合特征时无需准备原始 MRI、图谱、分通道缓存或单独的 FC 文件。可用 `--data-dir` 指定标签与划分目录。

## 使用已有的分通道缓存

```bash
python code/classify_single_combo.py
```

默认读取 `3dvit_features/` 中的 `smri_WM_vit_features.npy`、`fmri_ReHo_vit_features.npy`、`fmri_DC_vit_features.npy`，以及 `data/FC_matrices/` 中对应样本的 `.mat` 或 `.mat.gz` 文件（键 `FC`，大小 `116 × 116`）。使用 FC 时，需提供与特征行对齐的 `data/file_list.npy`。这些缓存中的 NIfTI 路径只用于提取样本 ID。按需使用 `--feature-dir` 和 `--channels` 切换目录与组合，例如 `--channels GM WM`。

[原仓库的 ViT 特征下载目录](https://drive.google.com/drive/folders/16lA0-9cZI0YnA7fe0Gz6856OewzHc9zi?usp=sharing)

## 输出

`results/` 下生成同名的 JSON 和 CSV 文件。JSON 保存五折的 ACC、AUC、F1、灵敏度、特异度、精确率、混淆矩阵和汇总统计。CSV 保存每一折在独立测试集上的样本 ID、真实标签、预测标签和 MDD 概率。
