# microModel

Cell image classification and self-supervised representation learning for fluorescence microscopy TIFFs.

## Overview

microModel trains [timm](https://github.com/huggingface/pytorch-image-models) backbones (ConvNeXt, ViT, etc.) on cropped single-cell fluorescence microscopy images. It supports:

- **Supervised classification** — Focal Loss training with mask-aware preprocessing
- **Self-supervised learning** — DINO (Distillation with No Labels) for representation learning
- **Inference** — Classify new images with per-class probabilities
- **Feature extraction** — Embed pre-classifier features (backbone-dependent dim) for downstream analysis

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `pyproject.toml` for full dependencies

```bash
pip install .
```

## Data preparation

Organise TIFF images as class-labeled subdirectories:

```
data_root/
├── class_A/
│   ├── image001.tiff
│   ├── image002.tiff
│   └── ...
├── class_B/
│   ├── image100.tiff
│   └── ...
└── class_C/
    └── ...
```

- Images can be single-channel (H×W), multi-channel (H×W×C), or already in channel-first (C×H×W).
- Channels are 1-indexed in config (e.g. `[1, 2, 5]` selects the 1st, 2nd, and 5th channels).
- Multiple root directories can be specified — same-named classes are merged.
- An optional separate `val_dir` with the same structure bypasses auto-splitting.

## Configuration

All training parameters are in a single YAML file in `configs/`. Two configs are provided:

| Config | Purpose |
|---|---|---|
| `configs/config_sl.yaml` | Supervised learning (ConvNeXt nano, 3 channels, 128px) |
| `configs/config_ssl_dinov3.yaml` | SSL DINO (ConvNeXt nano, 1 channel, 128px) |

Copy and edit the relevant config for your dataset. Key paths to update:

```yaml
data:
  train_dir:
    - "C:/path/to/your/data_root"   # update this; use forward slashes
  val_dir: null                      # or set to a separate validation directory
```

**Config gotchas:**
- `data.train_dir` must be a YAML list (prefix with `- `), even for a single directory.
- Backslashes are automatically normalized to forward slashes.
- On Windows, `num_workers` is auto-clamped to 0 (PyTorch spawn conflict). The
  config value is ignored on Windows.

## Supervised learning

### Train

```bash
micromodel train --mode sl --config configs/config_sl.yaml
```

After training, the script can optionally:
- Saves `best_model.pth` (best validation macro F1)
- Generates UMAP embedding visualization (`umap_val.pdf`)
- Saves confusion matrix (`confusion_matrix.pdf`) + classification report (`metrics.json`)
- Runs inference on the validation set (`val_predictions.csv`)

Pass `--post-analysis` to enable these post-training steps.

Outputs go to `runs/YYYYMMDD_HHMMSS/`.

### Key parameters in config_sl.yaml

| Parameter | Default | Description |
|---|---|---|
| `model.backbone` | `convnext_nano` | Any timm model name |
| `model.channel_indices` | `[1, 2, 5]` | 1-indexed channels from TIFF |
| `model.target_size` | `128` | Output spatial size after pad/resize |
| `model.pretrained` | `true` | Download pretrained weights |
| `model.weights_path` | `null` | Local `.safetensors`/`.pth` path (overrides `pretrained`) |
| `model.stem_key_override` | `null` | Override auto-detected stem conv key |
| `training.batch_size` | `64` | Mini-batch size |
| `training.epochs` | `2` | Max epochs (early stopping may finish early) |
| `training.lr` | `5e-4` | Base learning rate |
| `training.weight_decay` | `0.05` | AdamW weight decay |
| `training.warmup_epochs` | `5` | Linear warmup epochs |
| `training.focal_loss_gamma` | `2.0` | Focal Loss focusing parameter |
| `training.early_stopping_patience` | `5` | Patience for early stopping (macro F1) |
| `training.gradient_clip` | `3.0` | Max gradient norm (`null` to disable) |
| `data.train_val_split` | `0.8` | Train fraction when no `val_dir` set |
| `data.sample_n` | `400` | Max images per source (`null` = all) |
| `data.sample_by` | `"dataset"` | Sampling strategy: `"dataset"` (per root) or `"directory"` (per class subdir) |
| `data.preload` | `true` | Load all preprocessed images into RAM once |
| `data.preload_workers` | `8` | Thread pool workers for image preloading |
| `data.prefetch_factor` | `2` | DataLoader prefetch per worker (ignored when `num_workers=0`) |
| `strategy.cell_mask` | `false` | Extract binary mask from channel 0 |
| `strategy.scale_pad` | `true` | Scale-invariant resize (pad then resize) |
| `strategy.normalize` | `true` | Per-channel z-score normalization |
| `analysis.max_samples` | `10000` | Max samples for UMAP |
| `analysis.umap_n_neighbors` | `15` | UMAP n_neighbors |
| `analysis.umap_min_dist` | `0.1` | UMAP min_dist |

### Training flags

```bash
micromodel train --mode {sl,ssl} --config <path> [--post-analysis]
```

| Flag | Description |
|---|---|
| `--config <path>` | Path to YAML config file |
| `--mode {sl,ssl}` | Training mode (default: `sl`) |
| `--post-analysis` | Run UMAP, confusion matrix, predictions after training |

A `tqdm` progress bar (images/s, loss, ETA) is shown per epoch regardless of flags.

### SL augmentation parameters

```yaml
augmentation:
  rotation_degrees: 180
  hflip_prob: 0.5
  vflip_prob: 0.5
  scale_range: [0.75, 1.1]
  translate_range: [0.1, 0.1]
  brightness: 0.2
  contrast: 0.2
  apply_prob: 0.8
```

When `strategy.cell_mask` is `true`, the mask is re-derived from the augmented image (`img[0] > 0`) rather than warped alongside — same result, ~2× faster augmentation.

## Self-supervised learning (DINO)

### Train

```bash
micromodel train --mode ssl --config configs/config_ssl_dinov3.yaml
```

The SSL script:
- Trains a student–teacher pair with DINO loss
- Saves `best_model.pth` (best validation loss)
- Generates UMAP visualization of learned embeddings

### Pooling

SSL backbones produce feature maps via different pooling methods:

```yaml
training:
  pooling: gap          # CNN backbones (ConvNeXt): global average pooling
  # pooling: cls_token  # ViT backbones (vit_small_patch16_dinov3): CLS token
```

The configs already set this correctly for their respective backbones.

### SSL config additions

Beyond SL parameters, SSL configs have:

| Parameter | Default | Description |
|---|---|---|
| `head.out_dim` | `8192` | DINO prototype dimension |
| `head.head_hidden_dim` | `1024` | MLP hidden dimension |
| `head.teacher_temp` | `0.04` | Teacher softmax temperature |
| `head.student_temp` | `0.1` | Student softmax temperature |
| `head.center_momentum` | `0.9` | Center update momentum |
| `training.ema_momentum_start` | `0.996` | Teacher EMA momentum (start) |
| `training.ema_momentum_end` | `1.0` | Teacher EMA momentum (end) |
| `training.pooling` | `gap` | Feature pooling: `gap` or `cls_token` |

### SSL augmentation parameters

```yaml
augmentation:
  rotation_degrees: 360
  flip_prob: 0.5
  noise_std_range: [0.05, 0.15]
  blur_sigma_range: [0.1, 1.0]
  brightness_range: [0.85, 1.15]
  channel_dropout_prob: [0.15, 0.25]
```

## Inference

```bash
micromodel inference --checkpoint runs/20260617_165952/best_model.pth --input_type {single,dataset} [--input_dir <dir> | --filelist <file>] [--output_type class] [--output_type feature] [--output_type reduction] [--output_file results.db] [--checkpoint_type auto] [--reducer <pkl>] [--dataset_config <yaml>]
```

### Inference flags

| Flag | Description |
|---|---|
| `--checkpoint <pth>` | Path to `best_model.pth` (required) |
| `--input_type {single,dataset}` | Input mode (default: `single`) |
| `--input_dir <dir>` | Directory of TIFF images (single mode) |
| `--filelist <file>` | File with one image path per line (single mode) |
| `--dataset_config <yaml>` | Path to microProfiler PipelineConfig YAML (dataset mode) |
| `--output_file <path>` | Output path (`.db` or `.xlsx`; default: `results.db`) |
| `--output_type` | Repeatable: `class`, `feature`, `reduction` |
| `--reducer <pkl>` | Fitted UMAP `.pkl` (required if `reduction` in `--output_type`) |
| `--checkpoint_type {auto,sl,ssl}` | Override auto-detection (default: `auto`) |

### Single-image mode

```bash
micromodel inference --checkpoint runs/20260617_165952/best_model.pth --input_dir path/to/images --output_type class --output_file results.db
micromodel inference --checkpoint runs/20260617_165952/best_model.pth --filelist images.txt --output_type class --output_file results.xlsx
```

Output columns: `file`, `class`, `prob_<class_A>`, `prob_<class_B>`, ...

### Dataset mode (microProfiler)

Use `--input_type dataset` with a microProfiler `PipelineConfig` YAML to run inference on mask-defined cell objects:

```bash
micromodel inference --checkpoint runs/20260617_165952/best_model.pth --input_type dataset --dataset_config configs/config_inference_dataset.yaml --output_type class --output_file results.db
```

Each mask object produces one row, with per-class probabilities and metadata (well, field, mask name, cell ID).

### Feature extraction

```bash
micromodel inference --checkpoint runs/20260617_165952/best_model.pth --input_dir path/to/images --output_type feature --output_file results.db
```

To project into the same UMAP space as training:

```bash
micromodel inference --checkpoint runs/20260617_165952/best_model.pth --input_dir path/to/images --output_type reduction --reducer runs/20260617_165952/umap_reducer.pkl --output_file results.db
```

### Output format

Output files can be SQLite (`.db`) or Excel (`.xlsx`). Table/sheet names: `infer_class`, `infer_feature`, `infer_reduction`.

## Fine-tuning

To continue training from a previous checkpoint with a different dataset:

```yaml
model:
  finetune: runs/20260617_165952/best_model.pth
```

The model architecture, channel count, and target size are loaded from the checkpoint. Set `data.train_dir` to the new data. If classes differ, you'll be prompted to confirm the mapping.

## Backbone switching

Any [timm](https://github.com/huggingface/pytorch-image-models) model can be used:

```yaml
model:
  backbone: vit_small_patch16_dinov3
  target_size: 224
  channel_indices: [1, 2, 5]
  stem_key_override: "patch_embed.proj.weight"
```

For backbones with non-standard input channel counts, `stem_key_override` tells the code which weight tensor to adapt. Common values:

| Backbone | First conv key |
|---|---|
| ConvNeXt | `stem.0.weight` |
| ViT | `patch_embed.proj.weight` |
| ResNet | `conv1.weight` |
| EfficientNet | `conv_stem.weight` |

If omitted, the code auto-detects the stem key.

### ViT training considerations

| Parameter | ConvNeXt nano | ViT Small |
|---|---|---|
| `target_size` | 128 | 224 (patch-based, needs ≥224) |
| `batch_size` | 128 | 32–64 |
| `lr` | 1e-4 | 5e-4 to 1e-3 |
| `warmup_epochs` | 5 | 10–20 |
| `gradient_clip` | null (disabled) | 3.0 (recommended) |

## Preloading

For datasets smaller than ~10 GB with available RAM, enable preloading:

```yaml
data:
  preload: true
  preload_workers: 8   # thread pool for parallel I/O
```

This loads and preprocesses all images into memory at startup using a thread pool (`preload_workers`), eliminating TIFF I/O per epoch. Augmentation still runs per-epoch. With 64 GB RAM, this handles datasets up to ~40 GB without issue.

## Output structure

Each training run creates a timestamped directory:

```
runs/20260617_165952/
├── best_model.pth          # Best checkpoint (by val F1 for SL, val loss for SSL)
├── config.yaml             # Config copy for reproducibility
├── training_log.jsonl      # Per-epoch metrics
├── val_embeddings.npy      # Penultimate-layer embeddings
├── umap_val.pdf            # UMAP visualization
├── umap_reducer.pkl        # Fitted UMAP reducer (for projecting new data)
├── confusion_matrix.pdf    # [SL only] Confusion matrix
├── metrics.json            # [SL only] Classification report + best epoch stats
└── val_predictions.csv     # [SL only] Predictions on validation set
```

## File structure

```
microModel/
├── src/
│   └── microModel/              # Source package (import as `microModel.*`)
│       ├── __init__.py          # Public API re-exports, __version__
│       ├── cli.py               # CLI entry point (argparse subparsers)
│       ├── inference.py         # Inference engine + CLI
│       ├── train_sl.py          # Supervised learning loop
│       ├── train_ssl_dinov3.py  # SSL/DINO training loop
│       ├── dataset/
│       │   ├── __init__.py
│       │   ├── cell_dataset.py       # CellDataset (SL), CellAugmentation
│       │   ├── cell_dataset_ssl.py   # CellDatasetSSL, create_ssl_dataloader
│       │   ├── augmentation_ssl.py   # MultiCropAugmentation (DINO)
│       │   ├── preprocess.py         # load_and_preprocess, preprocess_array
│       │   └── _shared.py            # Preloading, DataLoader factory
│       ├── model/
│       │   ├── __init__.py
│       │   ├── model.py              # create_model, configure_optimizer
│       │   ├── heads.py              # DINOHead
│       │   └── losses.py             # FocalLoss, DINOLoss
│       └── utils/
│           ├── __init__.py
│           ├── config.py             # Config, TrainingLogger
│           ├── training.py           # build_cosine_warmup_scheduler, save_checkpoint, clip_gradients
│           ├── analysis.py           # run_sl_analysis, run_ssl_analysis, UMAP
│           └── utils.py              # natural_sort_key, can_compile
├── configs/                     # YAML configuration files
│   ├── config_sl.yaml
│   ├── config_ssl_dinov3.yaml
│   └── config_inference_dataset.yaml
├── runs/                        # Training output (gitignored)
├── test/                        # Test suite + test data
│   ├── run_all.py
│   ├── configs/
│   ├── testdata_single/
│   └── testdata_dataset/
├── pyproject.toml
└── README.md
```
