"""
demo.ipynb  (source — run: jupyter nbconvert --to notebook --execute notebooks/demo.ipynb)
==================================================================================
Final Year Project Demo Notebook
DOA Estimation: CNN vs MUSIC vs ESPRIT on Thinned Coprime Array

This notebook provides an interactive demonstration suitable for Final Presentation:
1. Array geometry visualisation
2. Signal simulation (choose K sources and SNR)
3. CNN inference (all 4 models)
4. Classical algorithms (MUSIC spectrum + ESPRIT)
5. Side-by-side comparison of estimated DOAs
6. SNR robustness curves
"""

# ── Cell 1: Imports and setup
import sys, os
sys.path.insert(0, os.path.abspath('..'))  # run from notebooks/ dir

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import torch
import warnings
warnings.filterwarnings('ignore')

from datasets.array_geometry  import get_tca_positions, get_steering_matrix
from datasets.generate_raw    import simulate_snapshot
from datasets.generate_cov    import simulate_cov
from models.resnext_doa       import ResNeXtDOA
from eval.compare_classical   import music, esprit, sample_covariance, simulate_signal
from eval.metrics             import compute_metrics

print("Imports OK")
positions = get_tca_positions(M=5, N=6)
print(f"TCA positions: {positions.tolist()}")
print(f"Number of sensors: {len(positions)}")
DOA_GRID = np.arange(-60, 61, 1)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Cell 2: Array geometry visualisation
fig, ax = plt.subplots(figsize=(11, 1.8))
sub1 = set(k * 6 for k in range(5))
sub2 = set(k * 5 for k in range(8))
for p in positions:
    c = 'purple' if (p in sub1 and p in sub2) else ('royalblue' if p in sub1 else 'tomato')
    ax.plot(p, 0, 'o', color=c, ms=13, mec='k')
    ax.text(p, -0.12, str(int(p)), ha='center', fontsize=9)
ax.set_xlim(-2, 37); ax.set_ylim(-0.4, 0.3)
ax.set_xlabel('Position (d = λ/2 units)'); ax.yaxis.set_visible(False)
ax.set_title('Thinned Coprime Array — M=5, N=6 — 12 Sensors')
plt.tight_layout(); plt.show()

# ── Cell 3: Load all 4 trained models
MODEL_CONFIGS = {
    'raw_t16': ('raw', 16),
    'raw_t32': ('raw', 32),
    'cov_t16': ('cov', 16),
    'cov_t32': ('cov', 32),
}
models = {}
for name, (itype, T) in MODEL_CONFIGS.items():
    ckpt_path = f'../results/checkpoints/{name}_best.pth'
    if not os.path.isfile(ckpt_path):
        print(f'  [skip] {ckpt_path} not found — train first')
        continue
    m = ResNeXtDOA(num_classes=121, input_type=itype, pretrained=False).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    models[name] = (m, itype, T)
    print(f'  Loaded: {name}')
print(f"\n{len(models)} models ready")

# ── Cell 4: Interactive demo — set parameters here
K_SOURCES = 3        # number of sources
SNR_DB    = 10.0     # SNR in dB
T_SNAP    = 16       # snapshots for simulation
RNG_SEED  = 0        # change for different scenarios

rng    = np.random.default_rng(RNG_SEED)
X, true_doas, K = simulate_signal(positions, T_SNAP, SNR_DB, K=K_SOURCES, rng=rng)
print(f"True DOAs: {np.sort(true_doas).tolist()} degrees")
R = sample_covariance(X)

# ── Cell 5: CNN Inference
print("=== CNN Predictions ===")
for name, (model, itype, T) in models.items():
    if itype == 'raw':
        x_in, _ = simulate_snapshot(positions, T, SNR_DB, rng)
    else:
        x_in, _ = simulate_cov(positions, T, SNR_DB, rng)
    with torch.no_grad():
        prob = model(torch.from_numpy(x_in[None]).to(DEVICE)).cpu().numpy()[0]
    # Find top-K peaks above threshold 0.3
    peaks = np.where(prob > 0.3)[0]
    est_doas = DOA_GRID[peaks]
    print(f"  {name:10s}: estimated DOAs = {sorted(est_doas.tolist())}  (K_est={len(est_doas)})")

# ── Cell 6: Classical algorithms
print("\n=== Classical Algorithms ===")
est_music  = music(R, K, positions)
est_esprit = esprit(R, K, positions)
print(f"  MUSIC : {sorted(est_music.tolist())}")
print(f"  ESPRIT: {sorted(est_esprit.tolist())}")
print(f"  True  : {sorted(true_doas.tolist())}")

# ── Cell 7: MUSIC spectrum plot
search_grid = np.arange(-90, 91, 0.1)
_, spectrum  = music(R, K, positions, search_grid=search_grid, return_spectrum=True)
spec_db = 10 * np.log10(spectrum / spectrum.max() + 1e-15)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(search_grid, spec_db, 'royalblue', lw=1.5, label='MUSIC Spectrum')
for td in true_doas:
    ax.axvline(td, color='tomato', lw=2, ls='--', label=f'True {td:.0f}°')
ax.set_xlabel('DOA (degrees)'); ax.set_ylabel('Normalised Power (dB)')
ax.set_title(f'MUSIC Spatial Spectrum  (K={K}, SNR={SNR_DB}dB, T={T_SNAP})')
ax.set_xlim(-90, 90); ax.grid(alpha=0.3)
ax.legend(fontsize=9); plt.tight_layout(); plt.show()

# ── Cell 8: Probability output comparison (all CNN models)
if models:
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 3 * len(models)), sharex=True)
    if len(models) == 1: axes = [axes]

    for ax, (name, (model, itype, T)) in zip(axes, models.items()):
        rng2 = np.random.default_rng(RNG_SEED)
        if itype == 'raw':
            x_in, _ = simulate_snapshot(positions, T, SNR_DB, rng2)
        else:
            x_in, _ = simulate_cov(positions, T, SNR_DB, rng2)
        with torch.no_grad():
            prob = model(torch.from_numpy(x_in[None]).to(DEVICE)).cpu().numpy()[0]

        ax.bar(DOA_GRID, prob, width=0.8, color='steelblue', alpha=0.7)
        for td in true_doas:
            ax.axvline(td, color='tomato', lw=2, ls='--')
        ax.set_ylabel('P(active)')
        ax.set_title(f'CNN Output: {name}')
        ax.set_ylim(0, 1); ax.axhline(0.3, color='gray', ls=':', lw=1)

    axes[-1].set_xlabel('DOA (degrees)')
    plt.suptitle(f'CNN Probability Outputs  (K={K}, SNR={SNR_DB}dB)', fontsize=13)
    plt.tight_layout(); plt.show()

print("\n[Demo complete]  Ready for presentation!")
