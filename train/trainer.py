"""
trainer.py
----------
Unified training loop for all four DOA model configurations.

Features:
  - BCELoss + AdamW optimizer
  - ReduceLROnPlateau scheduler
  - Early stopping (patience configurable)
  - TensorBoard logging (loss + accuracy per epoch)
  - Best-checkpoint saving
  - Resume from checkpoint (--resume flag)

Usage:
    python train/trainer.py --config configs/raw_t16.yaml
    python train/trainer.py --config configs/cov_t32.yaml --resume results/checkpoints/cov_t32_best.pth
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Project root on sys.path (run from project root)
sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.data_loader import get_dataloaders
from models.resnext_doa import ResNeXtDOA


# ── Metric helpers ─────────────────────────────────────────────────────────────

def binary_accuracy(preds: torch.Tensor, targets: torch.Tensor,
                    threshold: float = 0.5) -> float:
    """
    Element-wise binary accuracy across all classes.
    Matches the paper's accuracy definition (exact multi-label match per element).
    """
    pred_bin = (preds >= threshold).float()
    correct  = (pred_bin == targets).float()
    return correct.mean().item()


# ── Training helpers ───────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer=None,
    device: torch.device = torch.device('cpu'),
    is_train: bool = True,
) -> tuple:
    """
    Run one epoch (train or val).

    Returns
    -------
    avg_loss : float
    avg_acc  : float
    """
    model.train(is_train)
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for X, Y in tqdm(loader, leave=False, desc='train' if is_train else 'val'):
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            preds = model(X)             # (B, 121)
            loss  = criterion(preds, Y)

            if is_train:
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            total_acc  += binary_accuracy(preds.detach(), Y)
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ── Main training function ─────────────────────────────────────────────────────

def train(cfg: dict, resume_path: str = None) -> None:
    """
    Full training loop driven by a config dictionary.

    Parameters
    ----------
    cfg         : dict   Loaded YAML configuration
    resume_path : str    Optional path to checkpoint .pth to resume from
    """
    # ── Paths & Directories ────────────────────────────────────────────────────
    ckpt_dir = cfg['checkpoint_dir']
    log_dir  = cfg['log_dir']
    itype    = cfg['input_type']   # 'raw' or 'cov'
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # Derive a run name from cfg
    run_name = f"{itype}_t{cfg['T']}"
    ckpt_best = os.path.join(ckpt_dir, f'{run_name}_best.pth')

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[trainer] Device: {device}  |  Run: {run_name}')

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loaders = get_dataloaders(
        train_h5   = cfg['train_h5'],
        val_h5     = cfg['val_h5'],
        test_h5    = cfg['test_h5'],
        input_type = itype,
        batch_size = cfg['batch_size'],
        num_workers= cfg['num_workers'],
        pin_memory = (device.type == 'cuda'),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ResNeXtDOA(
        num_classes = cfg['num_classes'],
        input_type  = itype,
        pretrained  = True,
    ).to(device)
    print(f'  Parameters: {model.count_parameters():,}')

    # ── Loss / Optimiser / Scheduler ──────────────────────────────────────────
    criterion = nn.BCELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float('inf')

    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        print(f'  Resumed from epoch {start_epoch}  best_val_loss={best_val_loss:.4f}')

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=log_dir)

    # ── Training Loop ─────────────────────────────────────────────────────────
    patience_count = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(start_epoch, cfg['num_epochs']):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(
            model, loaders['train'], criterion, optimizer, device, is_train=True
        )
        vl_loss, vl_acc = run_epoch(
            model, loaders['val'], criterion, None, device, is_train=False
        )

        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        # TensorBoard
        writer.add_scalars('Loss', {'train': tr_loss, 'val': vl_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': tr_acc, 'val': vl_acc}, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        elapsed = time.time() - t0
        print(
            f'Epoch {epoch+1:3d}/{cfg["num_epochs"]}  '
            f'train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  '
            f'val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
            f't={elapsed:.1f}s'
        )

        # ── Save best checkpoint ───────────────────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_count = 0
            torch.save({
                'epoch'         : epoch,
                'model_state'   : model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss' : best_val_loss,
                'config'        : cfg,
                'history'       : history,
            }, ckpt_best)
            print(f'  -> Saved best checkpoint: {ckpt_best}')
        else:
            patience_count += 1
            if patience_count >= cfg['early_stop_patience']:
                print(f'  Early stopping at epoch {epoch+1}')
                break

    writer.close()
    print(f'\nTraining complete. Best val_loss={best_val_loss:.4f}')
    print(f'Checkpoint: {ckpt_best}')

    # Save training history for plotting
    import json
    hist_path = os.path.join(ckpt_dir, f'{run_name}_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'History saved: {hist_path}')


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DOA ResNeXt model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pth to resume from')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_path=args.resume)
