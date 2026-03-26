"""
run_smoke_test.py
-----------------
Quick smoke test for DOA-CNN-TCA-ResNeXt pipeline.
Run from project root: python smoke_test/run_smoke_test.py

Tests (in order):
  T1 - TCA array geometry
  T2 - Raw signal generation (small batch)
  T3 - Covariance matrix generation (small batch)
  T4 - HDF5 DataLoader
  T5 - ResNeXt-DOA model shape (raw input)
  T6 - ResNeXt-DOA model shape (cov input)
  T7 - Training loop (3 epochs, tiny dataset)
  T8 - MUSIC algorithm
  T9 - ESPRIT algorithm

Outputs:
  smoke_test/data/          -- tiny HDF5 files
  smoke_test/report.md      -- human-readable report
  smoke_test/report.json    -- machine-readable results
"""

import sys, os, time, json, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

RESULTS = []   # list of {id, name, status, msg, elapsed}

def run_test(test_id, name, fn):
    t0 = time.time()
    try:
        msg = fn()
        status = 'PASS'
        if msg is None:
            msg = 'OK'
    except Exception as e:
        status = 'FAIL'
        msg = traceback.format_exc()
    elapsed = time.time() - t0
    icon = '✅' if status == 'PASS' else '❌'
    print(f"  {icon} T{test_id} {name:45s} [{elapsed:.2f}s]")
    if status == 'FAIL':
        # Print first 3 lines of traceback
        for line in msg.strip().split('\n')[-4:]:
            print(f"      {line}")
    RESULTS.append({'id': test_id, 'name': name, 'status': status,
                    'msg': msg, 'elapsed': round(elapsed, 3)})
    return status == 'PASS'


# ── T1: TCA geometry ──────────────────────────────────────────────────────────
def t1_array_geometry():
    from datasets.array_geometry import get_tca_positions, get_steering_vector
    pos = get_tca_positions(M=5, N=6)
    expected = [0,5,6,10,12,15,20,25,36,42,48,54]
    assert list(pos.astype(int)) == expected, \
        f"Positions mismatch!\n  Got:      {list(pos.astype(int))}\n  Expected: {expected}"
    # Test steering vector
    a = get_steering_vector(0.0, pos)
    assert a.shape == (12,), f"Steering vector shape {a.shape}"
    assert np.allclose(np.abs(a), 1.0), "Steering vector magnitudes not unit"
    return f"Positions: {list(pos.astype(int))}"


# ── T2: Raw signal generation ─────────────────────────────────────────────────
def t2_generate_raw():
    from datasets.generate_raw import generate
    os.makedirs('smoke_test/data', exist_ok=True)
    generate(T=16, n_samples=500, snr_db=10,
             output_path='smoke_test/data/raw_t16_train.h5', n_workers=2)
    generate(T=16, n_samples=100, snr_db=10,
             output_path='smoke_test/data/raw_t16_val.h5',   n_workers=2)
    generate(T=16, n_samples=100, snr_db=10,
             output_path='smoke_test/data/raw_t16_test.h5',  n_workers=2)
    import h5py
    with h5py.File('smoke_test/data/raw_t16_train.h5', 'r') as f:
        shape = f['X'].shape
        assert shape == (500, 2, 12, 16), f"Wrong shape: {shape}"
        assert f['Y'].shape == (500, 121), f"Wrong label shape: {f['Y'].shape}"
    return f"HDF5 shape: {shape}, labels: (500,121)"


# ── T3: Covariance generation ─────────────────────────────────────────────────
def t3_generate_cov():
    from datasets.generate_cov import generate
    generate(T=16, n_samples=500, snr_db=10,
             output_path='smoke_test/data/cov_t16_train.h5', n_workers=2)
    generate(T=16, n_samples=100, snr_db=10,
             output_path='smoke_test/data/cov_t16_val.h5',   n_workers=2)
    generate(T=16, n_samples=100, snr_db=10,
             output_path='smoke_test/data/cov_t16_test.h5',  n_workers=2)
    import h5py
    with h5py.File('smoke_test/data/cov_t16_train.h5', 'r') as f:
        shape = f['X'].shape
        assert shape == (500, 2, 12, 12), f"Wrong shape: {shape}"
    return f"HDF5 shape: {shape}"


# ── T4: DataLoader ─────────────────────────────────────────────────────────────
def t4_dataloader():
    from datasets.data_loader import get_dataloaders
    loaders = get_dataloaders(
        train_h5='smoke_test/data/raw_t16_train.h5',
        val_h5  ='smoke_test/data/raw_t16_val.h5',
        test_h5 ='smoke_test/data/raw_t16_test.h5',
        input_type='raw', batch_size=32, num_workers=0, pin_memory=False,
    )
    X, Y = next(iter(loaders['train']))
    assert X.shape == (32, 2, 12, 16), f"Batch X shape: {X.shape}"
    assert Y.shape == (32, 121),       f"Batch Y shape: {Y.shape}"
    assert Y.sum(dim=1).min() >= 1,    "Some labels have no active class"
    return f"Batch X={tuple(X.shape)}, Y={tuple(Y.shape)}, avg_active_classes={Y.sum(dim=1).float().mean().item():.2f}"


# ── T5: Model shape - raw ─────────────────────────────────────────────────────
def t5_model_raw():
    import torch
    from models.resnext_doa import ResNeXtDOA
    model = ResNeXtDOA(num_classes=121, input_type='raw', pretrained=False)
    model.eval()
    for T in [16, 32]:
        x = torch.randn(2, 2, 12, T)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 121), f"T={T}: out shape {out.shape}"
        assert (out >= 0).all() and (out <= 1).all(), "Outputs not in [0,1]"
    n_params = model.count_parameters()
    return f"params={n_params:,}, output in [0,1] ✓  (tested T=16 and T=32)"


# ── T6: Model shape - cov ─────────────────────────────────────────────────────
def t6_model_cov():
    import torch
    from models.resnext_doa import ResNeXtDOA
    model = ResNeXtDOA(num_classes=121, input_type='cov', pretrained=False)
    model.eval()
    x = torch.randn(2, 2, 12, 12)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 121), f"out shape {out.shape}"
    return f"params={model.count_parameters():,}, cov input (2,2,12,12) -> (2,121) ✓"


# ── T7: Training loop (3 epochs) ──────────────────────────────────────────────
def t7_training():
    import yaml, torch
    # Minimal config for smoke test
    cfg = {
        'input_type': 'raw', 'T': 16, 'num_classes': 121,
        'train_h5': 'smoke_test/data/raw_t16_train.h5',
        'val_h5'  : 'smoke_test/data/raw_t16_val.h5',
        'test_h5' : 'smoke_test/data/raw_t16_test.h5',
        'batch_size': 32, 'num_epochs': 3, 'lr': 1e-3,
        'weight_decay': 1e-2, 'early_stop_patience': 5,
        'val_split': 0.1, 'num_workers': 0,
        'checkpoint_dir': 'smoke_test/checkpoints',
        'log_dir': 'smoke_test/logs/raw_t16',
    }
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    from train.trainer import train
    train(cfg)  # runs 3 epochs
    ckpt = 'smoke_test/checkpoints/raw_t16_best.pth'
    assert os.path.isfile(ckpt), "Checkpoint not saved!"
    size_mb = os.path.getsize(ckpt) / 1e6
    return f"3 epochs completed, checkpoint saved ({size_mb:.1f} MB)"


# ── T8: MUSIC algorithm ───────────────────────────────────────────────────────
def t8_music():
    from datasets.array_geometry import get_tca_positions, get_steering_matrix
    from eval.compare_classical import simulate_signal, sample_covariance, music

    positions = get_tca_positions()
    rng = np.random.default_rng(42)
    K = 3
    X, true_doas, _ = simulate_signal(positions, T=32, snr_db=10, K=K, rng=rng)
    R = sample_covariance(X)
    est = music(R, K, positions)
    assert len(est) == K, f"Expected {K} estimates, got {len(est)}"
    errs = []
    for td in sorted(true_doas):
        closest = min(abs(td - e) for e in est)
        errs.append(closest)
    mean_err = np.mean(errs)
    return (f"True: {sorted(true_doas.tolist())}  "
            f"Est: {sorted(est.tolist())}  "
            f"MeanErr={mean_err:.2f}deg")


# ── T9: ESPRIT algorithm ──────────────────────────────────────────────────────
def t9_esprit():
    from datasets.array_geometry import get_tca_positions
    from eval.compare_classical import simulate_signal, sample_covariance, esprit

    positions = get_tca_positions()
    rng = np.random.default_rng(7)
    K = 2
    X, true_doas, _ = simulate_signal(positions, T=32, snr_db=15, K=K, rng=rng)
    R = sample_covariance(X)
    est = esprit(R, K, positions)
    assert len(est) >= 1, "ESPRIT returned no estimates"
    return (f"True: {sorted(true_doas.tolist())}  "
            f"Est: {sorted(est[:K].tolist())}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  DOA-CNN-TCA-ResNeXt  --  Smoke Test")
    print("="*60 + "\n")

    run_test(1, "TCA array geometry",             t1_array_geometry)
    run_test(2, "Raw signal generation (500 samples)", t2_generate_raw)
    run_test(3, "Covariance generation (500 samples)", t3_generate_cov)
    run_test(4, "HDF5 DataLoader",                t4_dataloader)
    run_test(5, "ResNeXt-DOA model: raw input",   t5_model_raw)
    run_test(6, "ResNeXt-DOA model: cov input",   t6_model_cov)
    run_test(7, "Training loop (3 epochs)",        t7_training)
    run_test(8, "MUSIC algorithm",                 t8_music)
    run_test(9, "ESPRIT algorithm",                t9_esprit)

    # ── Summary ────────────────────────────────────────────────────────────────
    passed = sum(1 for r in RESULTS if r['status'] == 'PASS')
    total  = len(RESULTS)
    print("\n" + "="*60)
    print(f"  Result: {passed}/{total} tests passed")
    print("="*60 + "\n")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    import datetime
    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'passed': passed, 'total': total,
        'results': RESULTS,
    }
    os.makedirs('smoke_test', exist_ok=True)
    with open('smoke_test/report.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Save Markdown report ───────────────────────────────────────────────────
    lines = [
        "# Smoke Test Report",
        f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Result:** {passed}/{total} tests passed\n",
        "| # | Test | Status | Time | Details |",
        "|---|------|--------|------|---------|",
    ]
    for r in RESULTS:
        icon = '✅ PASS' if r['status'] == 'PASS' else '❌ FAIL'
        detail = r['msg'].strip().replace('\n', ' ')[:120]
        lines.append(f"| T{r['id']} | {r['name']} | {icon} | {r['elapsed']}s | {detail} |")

    if passed < total:
        lines += ["", "## Failures", ""]
        for r in RESULTS:
            if r['status'] == 'FAIL':
                lines += [f"### T{r['id']}: {r['name']}", "```", r['msg'], "```", ""]

    lines += [
        "",
        "## TCA Array Positions (verified against paper Fig.2)",
        "```",
        "X1 = {0,5,10,15,20,25}   (n*M, n=0..5)",
        "X2 = {6,12}              (m*N, m=1..2)",
        "X3 = {36,42,48,54}       ((m+M+1)*N, m=0..3)",
        "TCA = {0,5,6,10,12,15,20,25,36,42,48,54}  -> 12 sensors",
        "```",
        "",
        "## Environment",
        f"- Python: {sys.version.split()[0]}",
    ]
    try:
        import torch
        lines.append(f"- PyTorch: {torch.__version__}")
        lines.append(f"- CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"- GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    with open('smoke_test/report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"  Report saved: smoke_test/report.md")
    print(f"  JSON saved:   smoke_test/report.json\n")

    sys.exit(0 if passed == total else 1)
