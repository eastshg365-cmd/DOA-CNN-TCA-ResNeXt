# DOA-CNN-TCA-ResNeXt

<div align="center">

**Deep Learning Based Direction of Arrival Estimation**  
**基于深度学习的波达方向估计**

**Deep Learning Based Direction of Arrival Estimation**  
**基于深度学习的波达方向估计**

PolyU EIE4127 Final Year Project | PyTorch 2.11.0+cu126 | RTX 4090 D

[![GitHub](https://img.shields.io/badge/GitHub-eastshg365--cmd%2FDOA--CNN--TCA--ResNeXt-blue?logo=github)](https://github.com/eastshg365-cmd/DOA-CNN-TCA-ResNeXt)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange)
![Smoke Test](https://img.shields.io/badge/Smoke_Test-9%2F9_PASS-brightgreen)

</div>

---

## Overview | 项目概述

Reproduces the IEEE 2020 paper *"A Unified Approach for Target Direction Finding Based on Convolutional Neural Networks"* using a modified ResNeXt-50 on a Thinned Coprime Array (TCA).

复现 IEEE 2020 论文 — 统一框架下基于卷积神经网络的目标方向估计，使用改进版 ResNeXt-50 在稀疏互质阵列（TCA）上进行多标签 DOA 估计。

### Key Results | 关键结果

| Model | Input | T | Accuracy | Recall |
|-------|-------|---|----------|--------|
| CNN-RAW-T16 | Raw | 16 | TBD after training | TBD |
| CNN-RAW-T32 | Raw | 32 | TBD | TBD |
| CNN-COV-T16 | Cov | 16 | TBD | TBD |
| CNN-COV-T32 | Cov | 32 | TBD | TBD |

---

## Project Structure | 目录结构

```
DOA-CNN-TCA-ResNeXt/
├── configs/               # YAML hyperparameters for 4 model configs
├── data/                  # HDF5 datasets (generated, not committed)
├── datasets/
│   ├── array_geometry.py  # TCA sensor positions (M=5, N=6 → 12 sensors)
│   ├── generate_raw.py    # Raw signal generator (2, 12, T)
│   ├── generate_cov.py    # Covariance matrix generator (2, 12, 12)
│   └── data_loader.py     # PyTorch Dataset + DataLoader
├── models/
│   └── resnext_doa.py     # Modified ResNeXt-50: 2ch input, FC(121)+Sigmoid
├── train/
│   └── trainer.py         # BCELoss + AdamW + early stopping + TensorBoard
├── eval/
│   ├── metrics.py         # Accuracy / Precision / Recall / Specificity
│   ├── compare_classical.py  # MUSIC + ESPRIT benchmark
│   └── visualize.py       # Auto-generate all paper figures (Fig.4–8)
├── results/               # Auto-generated: checkpoints, tables, figures
├── notebooks/
│   └── demo.py            # Presentation demo script
├── README.md
├── requirements.txt
└── run_all.sh             # One-shot pipeline
```

---

## Quick Start | 快速开始

### 1. Install Dependencies | 安装依赖

```bash
pip install -r requirements.txt
```

### 2. One-Shot Run | 一键运行

```bash
# Full pipeline: generate data -> train all 4 models -> evaluate -> plot
bash run_all.sh
```
> ⚠️ Data generation takes ~3-5 hours (CPU multiprocessing). Training each model takes ~3-4 hours on RTX 4090.

### 3. Step-by-Step | 分步执行

```bash
# Generate datasets (adjust --samples for quick test, e.g. 10000)
python datasets/generate_raw.py --T 16 --samples 1600000 --snr 10 --out data/raw_t16_train.h5
python datasets/generate_raw.py --T 16 --samples  200000 --snr 10 --out data/raw_t16_val.h5
python datasets/generate_raw.py --T 16 --samples  200000 --snr 10 --out data/raw_t16_test.h5
# (repeat for T=32, and cov variants using generate_cov.py)

# Train one model
python train/trainer.py --config configs/raw_t16.yaml

# Evaluate and generate tables
python eval/metrics.py --config configs/raw_t16.yaml
python eval/metrics.py --config configs/raw_t16.yaml --all   # generate all tables

# Classical comparison (MUSIC + ESPRIT)
python eval/compare_classical.py --snr_range 0 20 --step 2

# Generate all figures
python eval/visualize.py --all

# Monitor training with TensorBoard
tensorboard --logdir results/logs
```

### 4. Demo Notebook | 演示

```bash
cd notebooks
jupyter notebook demo.py   # or: python demo.py
```

---

## Array Geometry | 阵列几何

**Thinned Coprime Array (TCA)**  M=5, N=6 → **12 sensors**

稀疏互质阵列，M=5，N=6，共 12 个传感器

```
Sub-array 1 (dense,  step M=5): {0, 5, 10, 15, 20, 25, 30, 35}
Sub-array 2 (sparse, step N=6): {0,  6, 12, 18, 24}
Union (12 unique positions)   : {0, 5, 6, 10, 12, 15, 18, 20, 24, 25, 30, 35}
```

> ⚠️ **Verify against paper Fig. 2 before training.** Only `array_geometry.py` needs updating if the paper uses a different TCA variant.  
> ⚠️ **训练前请对照论文 Fig.2 确认传感器位置。**若论文使用不同的 TCA 变体，只需修改 `array_geometry.py`。

---

## Model Architecture | 模型架构

Modified ResNeXt-50 (32×4d) following paper Table 1:

| Layer | Output Size | Details |
|-------|-------------|---------|
| Conv1 | 56×56 | 7×7, 64 ch, stride 2 — **2-channel input** |
| MaxPool | 28×28 | 3×3, stride 2 |
| Stage 1 | 28×28 | 3 × ResNeXt block (64-d, 32 groups) |
| Stage 2 | 14×14 | 4 × ResNeXt block (128-d, 32 groups) |
| Stage 3 | 7×7 | 6 × ResNeXt block (256-d, 32 groups) |
| Stage 4 | 4×4 | 3 × ResNeXt block (512-d, 32 groups) |
| GAP | 1×1 | Global Average Pooling |
| FC | 121 | Linear(2048→121) + **Sigmoid** |

**Loss:** BCELoss | **Optimizer:** AdamW (lr=1e-3, wd=1e-2) | **Labels:** 121-dim multi-hot

---

## Data Format | 数据格式

| Input Type | Shape | Description |
|------------|-------|-------------|
| Raw | (2, 12, T) | Real/Imag parts of received snapshots |
| Cov | (2, 12, 12) | Real/Imag parts of sample covariance matrix |
| Label | (121,) | Multi-hot: 1 at true DOA positions |

**DOA Grid 角度格网:** −60° to +60° with 1° step → 121 classes  
**Sources 信源数:** K = 1–16 (random per sample)

---

## Evaluation Metrics | 评估指标

Following the paper, element-wise over all 121 classes:

| Metric | Formula |
|--------|---------|
| Accuracy | (TP+TN)/(TP+FP+TN+FN) |
| Precision | TP/(TP+FP) |
| Recall | TP/(TP+FN) |
| Specificity | TN/(TN+FP) |

---

## Outputs | 输出

After running `run_all.sh`:

```
results/
├── checkpoints/
│   ├── raw_t16_best.pth     # ← upload to Google Drive
│   ├── raw_t32_best.pth
│   ├── cov_t16_best.pth
│   └── cov_t32_best.pth
├── tables/
│   ├── table_raw_t16.csv    # Table 2 (paper)
│   ├── table_raw_t32.csv    # Table 3
│   ├── table_cov_t16.csv    # Table 4
│   ├── table_cov_t32.csv    # Table 5
│   ├── table6_comparison.csv  # Extension Table 6
│   └── *.tex                # LaTeX versions
└── figures/
    ├── fig4_loss_curves.png
    ├── fig5_performance_vs_T.png
    ├── fig6_cnn_vs_classical.png
    ├── fig7_tca_layout.png
    └── fig8_music_spectrum.png
```

---

## Citation | 引用

If you use this code, please cite the original paper:

```bibtex
@article{doa_cnn_2020,
  title   = {A Unified Approach for Target Direction Finding Based on Convolutional Neural Networks},
  journal = {IEEE Transactions on ...},
  year    = {2020},
}
```

---

## Requirements | 环境要求

- Python 3.10+
- PyTorch 2.4 + CUDA 12.x
- RTX 4090 (or equivalent ≥ 16GB VRAM)
- ~150 GB disk for full 8M dataset

See `requirements.txt` for full dependency list.
