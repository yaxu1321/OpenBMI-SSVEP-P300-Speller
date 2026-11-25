# OpenBMI (Lee2019) 数据集目录说明

本目录用于缓存通过 MOABB 自动下载的 OpenBMI（Lee et al., *GigaScience*, 2019）双范式公开脑电数据：
- `moabb.datasets.Lee2019_SSVEP`（54 名被试 × 2 天 Session，4 分类 SSVEP：5.45 / 6.67 / 8.57 / 12.0 Hz）
- `moabb.datasets.Lee2019_ERP`（54 名被试 × 2 天 Session，6×6 字符矩阵 P300 拼写器，Target : NonTarget = 1 : 5）

运行 `python scripts/download_dataset.py` 或在 Google Colab 中运行 Notebook 时，原始数据会自动下载并缓存至 `data/mne_data/`（已加入 `.gitignore`，不随代码库提交）。
