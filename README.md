# MUSA 分类精简版

输入一个已经融合的 `final_features.npy`，运行原 MLP，输出五折模型在同一独立测试集上的 ACC 均值和标准差。

```bash
python -m pip install -r requirements.txt
python code/classify_single_combo.py --features final_features.npy
```

特征形状为 `[N,D]` 或 `[N,116,D]`。特征行序须与 `data/labels.npy` 一致，训练和测试索引从 `data/data_split.npy` 读取。使用可信的划分文件。

原默认组合按 `WM → FC → ReHo → DC` 的顺序沿最后一维拼接。[获取原始分通道特征与 FC 文件](https://drive.google.com/drive/folders/16lA0-9cZI0YnA7fe0Gz6856OewzHc9zi?usp=sharing)。
