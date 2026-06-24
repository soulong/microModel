import os
import time
import copy
import warnings
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")

import numpy as np
import torch
import torch.nn.functional as F
import timm
from tqdm import tqdm

from microModel.dataset import create_ssl_dataloader
from microModel.model import DINOHead, DINOLoss
from microModel.utils import can_compile, Config, TrainingLogger, run_ssl_analysis
from microModel.utils.training import build_cosine_warmup_scheduler, save_checkpoint, clip_gradients



def _make_get_global_repr(config):
    pooling = config.get("training", {}).get("pooling", "gap")
    if pooling == "cls_token":
        return lambda feats: feats[:, 0]
    return lambda feats: feats.mean(dim=[2, 3])


def build_student(config):
    backbone_name = config["model"]["backbone"]
    in_chans = len(config["model"]["channel_indices"])
    pretrained = config["model"].get("pretrained", True)
    weights_path = config["model"].get("weights_path")

    model = timm.create_model(
        backbone_name,
        pretrained=(pretrained and weights_path is None),
        in_chans=in_chans,
        num_classes=0,
    )

    if weights_path:
        if weights_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    return model


def update_teacher_ema(student_model, teacher_model, momentum):
    for param_s, param_t in zip(student_model.parameters(), teacher_model.parameters()):
        param_t.data = momentum * param_t.data + (1.0 - momentum) * param_s.data


def _compute_dino_loss(student_backbone, student_head, teacher_backbone, teacher_head,
                       views, num_global, out_dim, get_global_repr, loss_fn, device,
                       student_views_all=True):
    global_images = torch.cat(views[:num_global], dim=0).to(device)
    B = global_images.shape[0] // num_global

    with torch.no_grad():
        t_feats = teacher_backbone.forward_features(global_images)
        t_feats = get_global_repr(t_feats)
        t_logits = teacher_head(t_feats)

    student_views = views if student_views_all else views[:num_global]
    groups = defaultdict(list)
    for v in student_views:
        groups[v.shape[-2:]].append(v)
    s_logits_list = []
    for group in groups.values():
        stacked = torch.cat(group, dim=0).to(device)
        feats = student_backbone.forward_features(stacked)
        feats = get_global_repr(feats)
        logits = student_head(feats)
        s_logits_list.append(logits)
    s_logits = torch.cat(s_logits_list, dim=0)

    s_logits = s_logits.view(B, -1, out_dim)
    t_logits = t_logits.view(B, num_global, out_dim)

    dino_loss = 0.0
    for s in range(s_logits.shape[1]):
        for t in range(num_global):
            dino_loss += loss_fn(s_logits[:, s], t_logits[:, t])
    dino_loss = dino_loss / (s_logits.shape[1] * num_global)

    return dino_loss, t_logits


def train(config_path=None, device="auto", post_analysis=False):
    if config_path is None:
        config_path = "configs/config_ssl_dinov3.yaml"

    config = Config(config_path)

    torch.set_float32_matmul_precision('high')
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    pooling = config.get("training", {}).get("pooling", "gap")
    print(f"Pooling: {pooling}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config["output"]["run_dir"], timestamp)
    os.makedirs(run_dir, exist_ok=True)
    config.save(os.path.join(run_dir, "config.yaml"))

    train_loader, val_loader = create_ssl_dataloader(config)
    num_global = config["augmentation"]["num_global_crops"]
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    get_global_repr = _make_get_global_repr(config)

    student_backbone = build_student(config)
    in_dim = student_backbone.num_features
    out_dim = config["head"]["out_dim"]

    student_head = DINOHead(
        in_dim=in_dim,
        hidden_dim=config["head"].get("head_hidden_dim", 2048),
        bottleneck_dim=config["head"].get("head_bottleneck_dim", 256),
        out_dim=out_dim,
    )
    teacher_backbone = copy.deepcopy(student_backbone)
    teacher_head = copy.deepcopy(student_head)
    teacher_backbone.requires_grad_(False)
    teacher_head.requires_grad_(False)

    student_backbone = student_backbone.to(device)
    student_head = student_head.to(device)
    teacher_backbone = teacher_backbone.to(device)
    teacher_head = teacher_head.to(device)

    if can_compile(device):
        student_backbone = torch.compile(student_backbone)
        student_head = torch.compile(student_head)
        B = config["training"]["batch_size"]
        C = len(config["model"]["channel_indices"])
        T = config["model"]["target_size"]
        dummy = torch.zeros(B, C, T, T, device=device)
        student_backbone(dummy)
        student_head(student_backbone(dummy))

    loss_fn = DINOLoss(
        out_dim=out_dim,
        student_temp=config["head"]["student_temp"],
        teacher_temp=config["head"]["teacher_temp"],
        center_momentum=config["head"]["center_momentum"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(student_backbone.parameters()) + list(student_head.parameters()),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_epochs = config["training"]["epochs"]
    warmup_epochs = config["training"].get("warmup_epochs", 5)
    scheduler = build_cosine_warmup_scheduler(optimizer, total_epochs, warmup_epochs)

    ema_start = config["training"]["ema_momentum_start"]
    ema_end = config["training"]["ema_momentum_end"]
    best_loss = float("inf")
    logger = TrainingLogger(os.path.join(run_dir, "training_log.jsonl"))
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    print("Warmup...")
    t0 = time.time()
    student_backbone.train()
    student_head.train()
    teacher_backbone.eval()
    teacher_head.eval()
    warmup_batch = next(iter(train_loader))
    with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
        dino_loss, _ = _compute_dino_loss(
            student_backbone, student_head, teacher_backbone, teacher_head,
            warmup_batch["views"], num_global, out_dim, get_global_repr, loss_fn, device,
        )
    optimizer.zero_grad()
    if scaler:
        scaler.scale(dino_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        dino_loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    # Warmup validation
    student_backbone.eval()
    student_head.eval()
    with torch.no_grad():
        for val_batch in val_loader:
            views = val_batch["views"]
            global_images = torch.cat(views[:num_global], dim=0).to(device)
            t_feats = teacher_backbone.forward_features(global_images)
            t_feats = get_global_repr(t_feats)
            _ = teacher_head(t_feats)
            groups = defaultdict(list)
            for v in views[:num_global]:
                groups[v.shape[-2:]].append(v)
            for group in groups.values():
                stacked = torch.cat(group, dim=0).to(device)
                feats = student_backbone.forward_features(stacked)
                feats = get_global_repr(feats)
                _ = student_head(feats)
    student_backbone.train()
    student_head.train()
    print(f"Warmup done ({time.time() - t0:.1f}s)")

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        student_backbone.train()
        student_head.train()
        teacher_backbone.eval()
        teacher_head.eval()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                dino_loss, t_logits = _compute_dino_loss(
                    student_backbone, student_head, teacher_backbone, teacher_head,
                    batch["views"], num_global, out_dim, get_global_repr, loss_fn, device,
                )

            optimizer.zero_grad()
            if scaler:
                scaler.scale(dino_loss).backward()
            else:
                dino_loss.backward()

            if config["training"].get("gradient_clip") is not None:
                clip_gradients(
                    optimizer,
                    list(student_backbone.parameters()) + list(student_head.parameters()),
                    config["training"]["gradient_clip"],
                    scaler=scaler,
                )

            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            num_batches = batch_idx + 1
            progress = (epoch - 1) * len(train_loader) + num_batches - 1
            total_steps = total_epochs * len(train_loader)
            momentum = ema_start + (ema_end - ema_start) * min(progress / total_steps, 1.0)
            update_teacher_ema(student_backbone, teacher_backbone, momentum)
            update_teacher_ema(student_head, teacher_head, momentum)

            loss_fn.update_center(t_logits.detach().view(-1, out_dim))

            total_loss += dino_loss.item()
            pbar.set_postfix(loss=dino_loss.item())

        avg_loss = total_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - epoch_start

        student_backbone.eval()
        student_head.eval()
        val_loss_total = 0.0
        val_entropy_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                dino_loss, t_logits = _compute_dino_loss(
                    student_backbone, student_head, teacher_backbone, teacher_head,
                    batch["views"], num_global, out_dim, get_global_repr, loss_fn, device,
                    student_views_all=False,
                )

                t_centered = t_logits.reshape(-1, out_dim) - loss_fn.center
                t_probs = F.softmax(t_centered / loss_fn.teacher_temp, dim=-1)
                entropy = -(t_probs * torch.log(t_probs + 1e-8)).sum(dim=-1).mean()

                val_loss_total += dino_loss.item()
                val_entropy_total += entropy.item()
                val_batches += 1

        val_loss = val_loss_total / max(val_batches, 1)
        val_entropy = val_entropy_total / max(val_batches, 1)

        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_teacher_entropy": round(val_entropy, 4),
            "lr": round(current_lr, 8),
            "duration_s": round(duration, 1)
        }
        logger.log(log_entry)

        print(f"Epoch {epoch:3d}/{total_epochs} | train={avg_loss:.4f} val={val_loss:.4f} entropy={val_entropy:.4f} | lr={current_lr:.2e} | {duration:.1f}s")
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                "student_backbone": student_backbone.state_dict(),
                "student_head": student_head.state_dict(),
                "teacher_backbone": teacher_backbone.state_dict(),
                "teacher_head": teacher_head.state_dict(),
                "center": loss_fn.center.cpu(),
                "config": dict(config),
                "epoch": epoch,
                "val_loss": val_loss,
                "feature_dim": student_backbone.num_features,
            }
            save_checkpoint(checkpoint, run_dir, "val_loss", val_loss)

    if post_analysis:
        print(f"Post-training analysis...")
        run_ssl_analysis(student_backbone, lambda feats: student_head(get_global_repr(feats)),
                         val_loader, config, run_dir, device, num_global)

    print(f"Training complete. Best val_loss: {best_loss:.4f}")
    print(f"All outputs saved to {run_dir}")


if __name__ == "__main__":
    train()  # uses defaults (config_path=None → config_ssl_dinov3.yaml, device="auto")
