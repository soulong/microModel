import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Subset

from microModel.dataset import create_dataloaders_from_dirs
from microModel.model import create_model, configure_optimizer, load_finetune_model, FocalLoss
from microModel.utils import natural_sort_key, can_compile, Config, TrainingLogger, run_sl_analysis
from microModel.utils.training import save_checkpoint, clip_gradients
from microModel.inference import predict_on_images


def _confirm_finetune(old_names, new_names):
    old_set, new_set = set(old_names), set(new_names)
    added = sorted(new_set - old_set, key=natural_sort_key)
    removed = sorted(old_set - new_set, key=natural_sort_key)

    print(f"  Checkpoint classes ({len(old_names)}): {', '.join(old_names[:5])}...")
    print(f"  New data classes  ({len(new_names)}): {', '.join(new_names[:5])}...")

    if not added and not removed:
        msg = "Classes match. Proceed with fine-tuning?"
    elif added and not removed:
        msg = f"New classes: {', '.join(added)}. Init from first head row. Proceed?"
    elif removed and not added:
        msg = f"Extra classes in checkpoint: {', '.join(removed)}. Will drop them. Proceed?"
    else:
        msg = f"Added: {', '.join(added)} | Removed: {', '.join(removed)}. Proceed?"

    answer = input(f"  {msg} [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("Fine-tuning cancelled by user.")
        sys.exit(0)


def _get_val_paths(val_loader):
    ds = val_loader.dataset
    if hasattr(ds, "indices"):
        return [os.path.abspath(ds.dataset.samples[i][0]) for i in ds.indices]
    return [os.path.abspath(ds.samples[i][0]) for i in range(len(ds))]


def train(config_path=None, device="auto", post_analysis=False):
    if config_path is None:
        config_path = "configs/config_sl.yaml"

    config = Config(config_path)

    torch.set_float32_matmul_precision('high')
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config["output"]["run_dir"], timestamp)
    os.makedirs(run_dir, exist_ok=True)

    config.save(os.path.join(run_dir, "config.yaml"))

    finetune_path = config.get("model", {}).get("finetune")
    if finetune_path:
        print(f"Fine-tuning from: {finetune_path}")
        checkpoint = torch.load(finetune_path, map_location="cpu", weights_only=True)
        ckpt_label_map = checkpoint["label_map"]
        old_class_names = sorted(ckpt_label_map.keys(), key=natural_sort_key)

        config["model"] = checkpoint["config"]["model"]

        train_loader, val_loader, new_label_map = create_dataloaders_from_dirs(config)
        new_class_names = sorted(new_label_map.keys(), key=natural_sort_key)
        num_classes = len(new_label_map)

        print(f"Classes: {new_label_map}")
        print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

        _confirm_finetune(old_class_names, new_class_names)
        model, label_map = load_finetune_model(checkpoint, config, num_classes, new_class_names, device)
        del checkpoint
    else:
        train_loader, val_loader, label_map = create_dataloaders_from_dirs(config)
        num_classes = len(label_map)
        print(f"Classes: {label_map}")
        print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

        model = create_model(config, num_classes)
        model = model.to(device)

    if can_compile(device):
        model = torch.compile(model)
        C = len(config["model"]["channel_indices"])
        T = config["model"]["target_size"]
        model(torch.zeros(1, C, T, T, device=device))

    if isinstance(train_loader.dataset, Subset):
        samples = train_loader.dataset.dataset.samples
        indices = train_loader.dataset.indices
        all_labels = [samples[i][1] for i in indices]
    else:
        all_labels = [s[1] for s in train_loader.dataset.samples]
    class_counts = [0] * num_classes
    for label in all_labels:
        class_counts[label] += 1
    class_counts = np.array(class_counts, dtype=np.float32)
    alpha = (1.0 / (class_counts + 1e-8))
    alpha = alpha / alpha.sum() * num_classes
    alpha = torch.tensor(alpha, dtype=torch.float).to(device)

    criterion = FocalLoss(gamma=config["training"]["focal_loss_gamma"], alpha=alpha)
    optimizer, scheduler = configure_optimizer(model, config)

    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]
    logger = TrainingLogger(os.path.join(run_dir, "training_log.jsonl"))
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # Warmup: compile CUDA kernels on one batch (separate from training)
    print("Warmup...")
    t0 = time.time()
    model.train()
    warmup_batch = next(iter(train_loader))
    inputs, targets = warmup_batch
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
        logits = model(inputs)
        loss = criterion(logits, targets)
    if scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        for val_batch in val_loader:
            model(val_batch[0].to(device))
    model.train()
    print(f"Warmup done ({time.time() - t0:.1f}s)")

    for epoch in range(1, config["training"]["epochs"] + 1):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        train_all_preds = []
        train_all_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']}", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits, targets)

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if config["training"].get("gradient_clip") is not None:
                clip_gradients(optimizer, model.parameters(),
                               config["training"]["gradient_clip"], scaler=scaler)

            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            train_all_targets.append(targets.cpu().numpy())
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)
        train_all_preds = np.concatenate(train_all_preds)
        train_all_targets = np.concatenate(train_all_targets)
        train_acc = accuracy_score(train_all_targets, train_all_preds)
        train_f1 = f1_score(train_all_targets, train_all_preds, average="macro", zero_division=0)

        model.eval()
        val_loss = 0.0
        val_all_preds = []
        val_all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * inputs.size(0)
                val_all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                val_all_targets.append(targets.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_all_preds = np.concatenate(val_all_preds)
        val_all_targets = np.concatenate(val_all_targets)
        val_acc = accuracy_score(val_all_targets, val_all_preds)
        val_f1 = f1_score(val_all_targets, val_all_preds, average="macro", zero_division=0)
        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - epoch_start
        scheduler.step()

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "train_f1_macro": round(train_f1, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1_macro": round(val_f1, 4),
            "lr": round(current_lr, 8),
            "duration_s": round(duration, 1)
        }
        logger.log(log_entry)

        print(f"Epoch {epoch:3d}/{config['training']['epochs']} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
              f"lr={current_lr:.2e} | {duration:.1f}s")

        if val_f1 >= best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "label_map": label_map,
                "config": dict(config),
                "epoch": epoch,
                "val_f1": val_f1,
                "feature_dim": model.num_features,
            }
            save_checkpoint(checkpoint, run_dir, "val_f1", val_f1)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    print(f"Training complete. Best epoch: {best_epoch} (val_f1={best_f1:.4f})")

    if post_analysis:
        print(f"Post-training analysis...")
        run_sl_analysis(model, val_loader, label_map, config, run_dir, device,
                        best_epoch=best_epoch, best_val_f1_macro=best_f1)

        val_paths = _get_val_paths(val_loader)
        pred_csv = os.path.join(run_dir, "val_predictions.csv")
        predict_on_images(model, config, val_paths, label_map, device, pred_csv)

    print(f"All outputs saved to {run_dir}")


if __name__ == "__main__":
    train()  # uses defaults (config_path=None → config_sl.yaml, device="auto")
