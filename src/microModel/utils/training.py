"""Shared training utilities — warmup scheduler, checkpoint save, gradient clipping."""

import os
import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_cosine_warmup_scheduler(optimizer, total_epochs, warmup_epochs):
    """Cosine annealing with linear warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    total_epochs : int
        Total number of training epochs.
    warmup_epochs : int
        Number of linear warmup epochs (LR ramps from 0 to base LR).

    Returns
    -------
    LambdaLR
    """
    warmup_epochs = int(warmup_epochs)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(state, run_dir, metric_name, metric_value):
    """Save best-model checkpoint with a formatted message.

    Parameters
    ----------
    state : dict
        Checkpoint state dict to save.
    run_dir : str
        Output directory.
    metric_name : str
        Display name for the tracked metric (e.g. ``"val_f1"``).
    metric_value : float
        Current metric value.
    """
    torch.save(state, os.path.join(run_dir, "best_model.pth"))
    print(f"  -> New best model ({metric_name}={metric_value:.4f})")


def clip_gradients(optimizer, params, max_norm, scaler=None):
    """Clip gradient norms, unscaling first if AMP is active.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer (passed to ``scaler.unscale_`` when AMP active).
    params : iterable of torch.Tensor
        Model parameters to clip.
    max_norm : float
        Maximum gradient norm.
    scaler : torch.cuda.amp.GradScaler or None
        AMP scaler (if active, unscale before clip).
    """
    if scaler:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, max_norm)
