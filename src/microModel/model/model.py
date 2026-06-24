import torch
import torch.nn as nn
import timm

from microModel.utils import natural_sort_key
from microModel.utils.training import build_cosine_warmup_scheduler


def create_model(config, num_classes):
    model_name = config["model"]["backbone"]
    weights_path = config["model"].get("weights_path")
    pretrained = config["model"].get("pretrained", True)
    channel_indices = config["model"]["channel_indices"]
    in_chans = len(channel_indices)
    stem_key_override = config["model"].get("stem_key_override")

    model = timm.create_model(
        model_name,
        pretrained=(pretrained and weights_path is None),
        in_chans=3,
        num_classes=0,
    )

    if weights_path:
        _load_weights_from_file(model, weights_path)

    if in_chans != 3:
        model = _adapt_stem_channels(model, in_chans, stem_key_override)

    model.reset_classifier(num_classes)
    return model


def _load_weights_from_file(model, weights_path):
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)


def _find_stem_key(state_dict):
    for key, tensor in state_dict.items():
        if tensor.dim() == 4 and tensor.shape[1] == 3:
            return key
    raise ValueError(
        "Could not auto-detect stem conv layer (no 4D weight with 3 input channels found). "
        "Set model.stem_key_override in config to the first conv layer key, e.g.:\n"
        '  ConvNeXt: "stem.0.weight"\n'
        '  ViT:      "patch_embed.proj.weight"\n'
        '  ResNet:   "conv1.weight"\n'
        '  EffNet:   "conv_stem.weight"'
    )


def _adapt_stem_channels(model, target_in_chans, stem_key_override=None):
    old_sd = model.state_dict()

    if stem_key_override:
        stem_key = stem_key_override
        if stem_key not in old_sd:
            raise KeyError(
                f"stem_key_override '{stem_key}' not found in state_dict. "
                f"Available keys with 4D weight: {[k for k, v in old_sd.items() if v.dim() == 4]}"
            )
    else:
        stem_key = _find_stem_key(old_sd)

    W = old_sd[stem_key]
    W_new = W.mean(dim=1, keepdim=True).expand(-1, target_in_chans, -1, -1).contiguous()

    model_name = model.default_cfg["architecture"]
    new_model = timm.create_model(model_name, pretrained=False, in_chans=target_in_chans, num_classes=0)
    new_sd = new_model.state_dict()
    for k in new_sd:
        if k in old_sd and old_sd[k].shape == new_sd[k].shape:
            new_sd[k] = old_sd[k]
    new_sd[stem_key] = W_new
    new_model.load_state_dict(new_sd)
    return new_model


def load_finetune_model(checkpoint, config, num_classes, new_class_names, device):
    ckpt_label_map = checkpoint["label_map"]
    old_class_names = sorted(ckpt_label_map.keys(), key=natural_sort_key)

    model = create_model(config, num_classes)
    state_dict = checkpoint["model_state_dict"]

    old_head_w = state_dict.get("head.fc.weight")
    old_head_b = state_dict.get("head.fc.bias")

    for k in list(state_dict.keys()):
        if "head.fc" in k:
            state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)

    if old_class_names != new_class_names and old_head_w is not None:
        first_idx = ckpt_label_map[old_class_names[0]]
        with torch.no_grad():
            for ni, name in enumerate(new_class_names):
                if name in ckpt_label_map:
                    oi = ckpt_label_map[name]
                    model.head.fc.weight[ni] = old_head_w[oi]
                    model.head.fc.bias[ni] = old_head_b[oi]
                else:
                    model.head.fc.weight[ni] = old_head_w[first_idx]
                    model.head.fc.bias[ni] = old_head_b[first_idx]

    label_map = {name: i for i, name in enumerate(new_class_names)}
    return model.to(device), label_map


def configure_optimizer(model, config):
    lr = config["training"]["lr"]
    wd = config["training"]["weight_decay"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    total_epochs = config["training"]["epochs"]
    warmup = config["training"].get("warmup_epochs", 5)
    scheduler = build_cosine_warmup_scheduler(optimizer, total_epochs, warmup)
    return optimizer, scheduler
