# Smoke Test Report
**Date:** 2026-03-24 19:04:26  
**Result:** 9/9 tests passed

| # | Test | Status | Time | Details |
|---|------|--------|------|---------|
| T1 | TCA array geometry | ✅ PASS | 0.199s | Positions: [0, 5, 6, 10, 12, 15, 20, 25, 36, 42, 48, 54] |
| T2 | Raw signal generation (500 samples) | ✅ PASS | 1.031s | HDF5 shape: (500, 2, 12, 16), labels: (500,121) |
| T3 | Covariance generation (500 samples) | ✅ PASS | 1.005s | HDF5 shape: (500, 2, 12, 12) |
| T4 | HDF5 DataLoader | ✅ PASS | 1.198s | Batch X=(32, 2, 12, 16), Y=(32, 121), avg_active_classes=9.50 |
| T5 | ResNeXt-DOA model: raw input | ✅ PASS | 0.834s | params=23,224,697, output in [0,1] ✓  (tested T=16 and T=32) |
| T6 | ResNeXt-DOA model: cov input | ✅ PASS | 0.197s | params=23,224,697, cov input (2,2,12,12) -> (2,121) ✓ |
| T7 | Training loop (3 epochs) | ✅ PASS | 2.173s | 3 epochs completed, checkpoint saved (279.2 MB) |
| T8 | MUSIC algorithm | ✅ PASS | 0.396s | True: [-50, 19, 32]  Est: [-50, 19, 32]  MeanErr=0.00deg |
| T9 | ESPRIT algorithm | ✅ PASS | 0.0s | True: [15, 53]  Est: [-7.77390068046688, -0.4755863050942857] |

## TCA Array Positions (verified against paper Fig.2)
```
X1 = {0,5,10,15,20,25}   (n*M, n=0..5)
X2 = {6,12}              (m*N, m=1..2)
X3 = {36,42,48,54}       ((m+M+1)*N, m=0..3)
TCA = {0,5,6,10,12,15,20,25,36,42,48,54}  -> 12 sensors
```

## Environment
- Python: 3.10.6
- PyTorch: 2.11.0+cu126
- CUDA available: True
- GPU: NVIDIA GeForce RTX 4090 D