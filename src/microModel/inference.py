"""Unified inference — class prediction, feature extraction, UMAP reduction.

Replaces predict.py and extract_features.py.
"""

from __future__ import annotations

import os
import csv
import pickle
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

import timm

from microModel.dataset import load_and_preprocess, preprocess_array
from microModel.model import create_model
from microModel.utils import _init_plotting

__all__ = [
    "InferenceEngine",
    "_detect_checkpoint_type",
    "preprocess_array",
    "objects_from_mask",
]

# ── Forward adapters ───────────────────────────────────────────────────────

class _ForwardAdapter:
    """Base for forward-pass adapters (SL and SSL)."""

    @torch.no_grad()
    def forward(self, tensor):
        raise NotImplementedError

    @torch.no_grad()
    def forward_batch(self, tensor):
        return self.forward(tensor)

    @property
    def supported_output_types(self):
        return ["class", "feature", "reduction"]


class _SLForwardAdapter(_ForwardAdapter):
    def __init__(self, model, config, label_map, device=None):
        self.model = model.eval()
        self.config = config
        self.label_map = label_map
        self.idx_to_class = {v: k for k, v in label_map.items()}
        self.num_classes = len(label_map)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = ckpt["config"]
        label_map = ckpt["label_map"]
        model = create_model(config, len(label_map))
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, config, label_map, device=device)

    @torch.no_grad()
    def forward(self, tensor):
        feats = self.model.forward_features(tensor)
        logits = self.model.forward_head(feats, pre_logits=False)
        features = self.model.forward_head(feats, pre_logits=True)
        return logits.cpu().numpy(), features.cpu().numpy()


class _SSLForwardAdapter(_ForwardAdapter):
    def __init__(self, backbone, config, pool_fn=None, feature_dim=None, device=None):
        self.backbone = backbone.eval()
        self.config = config
        self.label_map = None
        self.idx_to_class = {}
        self.num_classes = 0
        pooling = config.get("training", {}).get("pooling", "gap")
        self._pool = pool_fn or (
            (lambda feats: feats[:, 0]) if pooling == "cls_token"
            else (lambda feats: feats.mean(dim=[2, 3]))
        )
        self.feature_dim = feature_dim or backbone.num_features
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.backbone = self.backbone.to(self.device)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, use_teacher=True):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = ckpt["config"]
        in_chans = len(config["model"]["channel_indices"])
        backbone = timm.create_model(
            config["model"]["backbone"],
            pretrained=(config["model"].get("pretrained", True)
                        and config["model"].get("weights_path") is None),
            in_chans=in_chans,
            num_classes=0,
        )
        weights_path = config["model"].get("weights_path")
        if weights_path:
            if weights_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                sd = load_file(weights_path)
            else:
                sd = torch.load(weights_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(sd, strict=False)

        key = "teacher_backbone" if use_teacher else "student_backbone"
        backbone.load_state_dict(ckpt[key])
        return cls(backbone, config, device=device)

    @torch.no_grad()
    def forward(self, tensor):
        feats = self.backbone.forward_features(tensor)
        features = self._pool(feats)
        return None, features.cpu().numpy()

    @property
    def supported_output_types(self):
        return ["feature", "reduction"]


# ── Core inference engine ─────────────────────────────────────────────────

class InferenceEngine:
    """Model wrapper for forward passes with class prediction, feature
    extraction, and per-mask-object processing.

    Parameters
    ----------
    checkpoint_or_adapter : str or _ForwardAdapter
        Path to a checkpoint (auto-detects SL/SSL) or a pre-built adapter.
    device : torch.device or None
    """

    def __init__(self, checkpoint_or_adapter, device=None):
        if isinstance(checkpoint_or_adapter, _ForwardAdapter):
            self._adapter = checkpoint_or_adapter
        else:
            # Legacy: load from checkpoint path, auto-detect type
            ckpt_type = _detect_checkpoint_type(str(checkpoint_or_adapter))
            if ckpt_type == "sl":
                self._adapter = _SLForwardAdapter.from_checkpoint(
                    checkpoint_or_adapter, device
                )
            else:
                self._adapter = _SSLForwardAdapter.from_checkpoint(
                    checkpoint_or_adapter, device
                )
        self.device = device or self._adapter.device
        self.config = self._adapter.config
        self.label_map = self._adapter.label_map
        self.idx_to_class = self._adapter.idx_to_class
        self.num_classes = self._adapter.num_classes

    @torch.no_grad()
    def _forward(self, tensor):
        return self._adapter.forward(tensor)

    @torch.no_grad()
    def forward_batch(self, tensor):
        return self._adapter.forward_batch(tensor)

    @classmethod
    def from_model(cls, model, config, label_map, device=None):
        """Create engine from an already-loaded SL model (no checkpoint path).

        Parameters
        ----------
        model : torch.nn.Module
            Loaded model in eval mode.
        config : dict
            Config dict (e.g. from a previously loaded checkpoint).
        label_map : dict[str, int]
            Class name to index mapping.
        device : torch.device or None
        """
        return cls(_SLForwardAdapter(model, config, label_map, device=device))

    @classmethod
    def from_backbone(cls, backbone, config, pool_fn=None, feature_dim=None, device=None):
        """Create engine from an already-loaded SSL backbone (no checkpoint path).

        Parameters
        ----------
        backbone : torch.nn.Module
            Loaded backbone in eval mode.
        config : dict
            Config dict (e.g. from a previously loaded checkpoint).
        pool_fn : callable or None
            Pooling function (default: derived from ``training.pooling`` in config).
        feature_dim : int or None
            Feature dimension (default: ``backbone.num_features``).
        device : torch.device or None
        """
        return cls(_SSLForwardAdapter(backbone, config, pool_fn, feature_dim, device=device))

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None):
        """Load engine from a checkpoint, auto-detecting SL/SSL."""
        ckpt_type = _detect_checkpoint_type(checkpoint_path)
        if ckpt_type == "sl":
            adapter = _SLForwardAdapter.from_checkpoint(checkpoint_path, device)
        else:
            adapter = _SSLForwardAdapter.from_checkpoint(checkpoint_path, device)
        return cls(adapter, device)

    # ── single mode ───────────────────────────────────────────────────

    def process_single(self, image_paths, progress_callback=None):
        """Process individual cropped TIFF files.

        Parameters
        ----------
        image_paths : list of str
        progress_callback : callable or None
            Called as ``(message, current, total)`` after each image.

        Yields
        ------
        dict
            Keys: ``file``, ``logits``, ``features``.
        """
        total = len(image_paths)
        for i, path in enumerate(image_paths):
            if progress_callback:
                progress_callback(f"Image {i + 1}/{total}", i + 1, total)
            try:
                img, _ = load_and_preprocess(path, self.config)
            except ValueError as e:
                print(f"  Skipping {path}: {e}")
                continue
            tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
            logits, features = self._adapter.forward(tensor)
            yield {
                "file": os.path.abspath(path),
                "logits": logits,
                "features": features,
            }

    # ── dataset mode — in microProfiler.profiling.inference_dataset ─────


# ── Checkpoint type detection ─────────────────────────────────────────────

def _detect_checkpoint_type(checkpoint_path):
    """Return ``\"sl\"`` or ``\"ssl\"`` based on checkpoint keys."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in ckpt:
        return "sl"
    if "student_backbone" in ckpt:
        return "ssl"
    raise KeyError(
        f"Cannot determine checkpoint type from {checkpoint_path}. "
        "SL checkpoints must contain 'model_state_dict'; "
        "SSL checkpoints must contain 'student_backbone'."
    )

def objects_from_mask(mask_array):
    """Return list of (cell_id, bbox_slice) for every object in *mask_array*.

    Binary masks (only 0 and 1) are first split into connected components
    via ``ndi.label``.  Pre-labeled masks (multiple non-zero values) use
    the existing label values directly.
    """
    unique_vals = np.unique(mask_array)
    unique_vals = unique_vals[unique_vals != 0]
    if len(unique_vals) == 0:
        return []

    # Single non-zero value → binary mask, split into connected components
    if len(unique_vals) == 1:
        binary = (mask_array == unique_vals[0])
        labeled, n_features = ndi.label(binary)
        if n_features == 0:
            return []
        slices = ndi.find_objects(labeled)
        results = []
        for cell_id in range(1, n_features + 1):
            obj_slice = slices[cell_id - 1]
            if obj_slice is None:
                continue
            results.append((cell_id, obj_slice))
        return results

    # Pre-labeled mask → use existing labels
    results = []
    for cell_id in unique_vals:
        obj_mask = (mask_array == cell_id)
        bbox = ndi.find_objects(obj_mask.astype(np.int32))
        if bbox is None or len(bbox) == 0 or bbox[0] is None:
            continue
        results.append((int(cell_id), bbox[0]))
    return results


# ── Data helpers ──────────────────────────────────────────────────────────

def _get_image_paths(args):
    """Collect image paths from ``--input_dir`` or ``--filelist``."""
    if args.filelist:
        with open(args.filelist) as f:
            return [line.strip() for line in f if line.strip()]
    paths = []
    for root, _dirs, files in os.walk(args.input_dir):
        for fname in sorted(files):
            if fname.lower().endswith((".tiff", ".tif")):
                paths.append(os.path.join(root, fname))
    return paths


# ── Inference helpers (single-image mode for CLI) ──────────────────────


def _plot_umap(reduction_rows, class_rows, output_path):
    """Save ``umap_plot.pdf`` alongside the output file."""
    _init_plotting()
    import matplotlib.pyplot as plt

    coords = np.array([[r["umap_1"], r["umap_2"]] for r in reduction_rows])
    output_path = Path(output_path)
    plot_path = output_path.with_name(output_path.stem + "_umap.pdf")

    plt.figure(figsize=(8, 6))
    if class_rows is not None:
        preds = [r["class"] for r in class_rows]
        classes = sorted(set(preds))
        cmap = plt.cm.tab10
        for i, cls_name in enumerate(classes):
            mask = np.array([p == cls_name for p in preds])
            if not mask.any():
                continue
            plt.scatter(coords[mask, 0], coords[mask, 1],
                        c=[cmap(i % 10)], label=cls_name,
                        alpha=0.7, s=10)
        plt.legend()
    else:
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=10)

    plt.title("UMAP Projection")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"  UMAP plot: {plot_path}")


# ── Output writers ────────────────────────────────────────────────────────

def write_results(class_rows, feature_rows, reduction_rows, output_path,
                  if_exists="replace"):
    """Write collected results to ``.db`` (SQLite) or ``.xlsx`` (Excel).

    Parameters
    ----------
    class_rows : list[dict] or None
    feature_rows : list[dict] or None
    reduction_rows : list[dict] or None
    output_path : str | Path
        Must end in ``.db`` or ``.xlsx``.
    if_exists : str
        ``"replace"`` or ``"append"`` — passed to SQLite ``save_table``.
    """
    if all(x is None for x in (class_rows, feature_rows, reduction_rows)):
        print("No results to write.")
        return

    output_path = Path(output_path)
    ext = output_path.suffix.lower()

    if ext == ".db":
        _write_sqlite(class_rows, feature_rows, reduction_rows, output_path,
                      if_exists=if_exists)
    elif ext == ".xlsx":
        _write_xlsx(class_rows, feature_rows, reduction_rows, output_path)
    else:
        raise ValueError(
            f"Unsupported output extension '{ext}'.  Use .db or .xlsx."
        )


def _write_sqlite(class_rows, feature_rows, reduction_rows, db_path,
                  if_exists="replace"):
    from microProfiler import Database
    db = Database(db_path)
    try:
        if class_rows is not None:
            df = pd.DataFrame(class_rows)
            db.save_table(df, "infer_class", if_exists=if_exists)
            print(f"  infer_class  · {len(df)} rows")
        if feature_rows is not None:
            df = pd.DataFrame(feature_rows)
            db.save_table(df, "infer_feature", if_exists=if_exists)
            print(f"  infer_feature  · {len(df)} rows")
        if reduction_rows is not None:
            df = pd.DataFrame(reduction_rows)
            db.save_table(df, "infer_reduction", if_exists=if_exists)
            print(f"  infer_reduction  · {len(df)} rows")
    finally:
        db.close()
    print(f"Results saved to {db_path}")


def _write_xlsx(class_rows, feature_rows, reduction_rows, xlsx_path):
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        if class_rows is not None:
            df = pd.DataFrame(class_rows)
            df.to_excel(writer, sheet_name="infer_class", index=False)
            print(f"  infer_class  · {len(df)} rows")
        if feature_rows is not None:
            df = pd.DataFrame(feature_rows)
            df.to_excel(writer, sheet_name="infer_feature", index=False)
            print(f"  infer_feature  · {len(df)} rows")
        if reduction_rows is not None:
            df = pd.DataFrame(reduction_rows)
            df.to_excel(writer, sheet_name="infer_reduction", index=False)
            print(f"  infer_reduction  · {len(df)} rows")
    print(f"Results saved to {xlsx_path}")


# ── Backward-compatible entry point for train_sl.py ───────────────────────

def predict_on_images(model, config, image_paths, label_map, device, output_path):
    """Legacy CSV writer for post-training validation predictions.

    This function preserves the original signature from the old
    ``predict.py`` so that ``train_sl.py`` can call it without changes.
    New code should use ``inference.py`` CLI instead.
    """
    idx_to_class = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    model.eval()

    all_probs = []
    all_preds = []
    valid_paths = []

    for img_path in image_paths:
        try:
            img, _ = load_and_preprocess(img_path, config)
        except ValueError as e:
            print(f"  Skipping {img_path}: {e}")
            continue
        img_t = torch.from_numpy(img).float().unsqueeze(0).to(device)
        logits = model(img_t)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().detach().numpy()
        all_probs.append(probs)
        all_preds.append(idx_to_class[int(probs.argmax())])
        valid_paths.append(os.path.abspath(img_path))

    with open(output_path, "w", newline="") as f:
        class_names = [idx_to_class[i] for i in range(num_classes)]
        writer = csv.writer(f)
        writer.writerow(["file", "class"] + [f"prob_{name}" for name in class_names])
        for path, pred, probs in zip(valid_paths, all_preds, all_probs):
            writer.writerow([path, pred] + [f"{p:.6f}" for p in probs])
    print(f"Predictions saved to {output_path}")


# ── CLI entry point ───────────────────────────────────────────────────────

def main(args=None):
    """Run inference.

    Parameters
    ----------
    args : argparse.Namespace or None
        If None, parse from ``sys.argv`` (backward-compatible standalone entry).
    """
    parser = argparse.ArgumentParser(
        description="Unified inference — class prediction, feature extraction, "
                    "and UMAP reduction.",
        add_help=args is None,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--input_dir", default=None,
                        help="Directory of TIFF images")
    parser.add_argument("--filelist", default=None,
                        help="File with one image path per line")
    parser.add_argument("--output_file", default="results.db",
                        help="Output path — .db for SQLite, .xlsx for Excel "
                             "(default: results.db)")
    parser.add_argument("--output_type", action="append",
                        choices=["class", "feature", "reduction"],
                        help="Output type(s) to produce (repeatable)")
    parser.add_argument("--reducer", default=None,
                        help="Fitted UMAP reducer .pkl "
                             "(required if reduction in --output_type)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device (default: auto)")
    parser.add_argument("--checkpoint_type", default="auto",
                        choices=["auto", "sl", "ssl"],
                        help="Checkpoint type (default: auto-detect)")

    if args is None:
        args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────
    if not args.input_dir and not args.filelist:
        parser.error("requires --input_dir or --filelist")
    if not args.output_type:
        parser.error("at least one --output_type is required")

    if "reduction" in args.output_type and not args.reducer:
        parser.error("--reducer is required when reduction in --output_type")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Run ──────────────────────────────────────────────────────────
    engine = InferenceEngine.from_checkpoint(args.checkpoint, device=device)
    ckpt_type = "sl" if engine.label_map is not None else "ssl"
    print(f"Checkpoint type: {ckpt_type}")

    image_paths = _get_image_paths(args)
    if not image_paths:
        print("No TIFF images found.")
        return
    print(f"Found {len(image_paths)} images")

    class_rows = []
    feature_rows = []
    reduction_buf = []

    class_names = None
    if engine.label_map is not None:
        class_names = [engine.idx_to_class[i] for i in range(engine.num_classes)]

    for result in engine.process_single(image_paths):
        logits = result.pop("logits")
        features = result.pop("features")

        if "class" in args.output_type and logits is not None:
            probs = F.softmax(torch.from_numpy(logits), dim=1).squeeze(0).numpy()
            pred = int(probs.argmax())
            row = dict(result)
            row["class"] = engine.idx_to_class[pred]
            for i, name in enumerate(class_names):
                row[f"prob_{name}"] = float(f"{probs[i]:.6f}")
            class_rows.append(row)

        if "feature" in args.output_type:
            row = dict(result)
            feat = features.squeeze(0)
            for i in range(feat.shape[0]):
                row[f"feat_{i}"] = float(feat[i])
            feature_rows.append(row)

        if "reduction" in args.output_type:
            reduction_buf.append((dict(result), features.squeeze(0)))

    reduction_rows = None
    if "reduction" in args.output_type and reduction_buf:
        print(f"Reducing {len(reduction_buf)} feature vectors to 2-D …")
        with open(args.reducer, "rb") as f:
            rd = pickle.load(f)
        all_feats = np.stack([r[1] for r in reduction_buf], axis=0)
        coords = rd.transform(all_feats)
        reduction_rows = []
        for (meta, _feat), coord in zip(reduction_buf, coords):
            row = dict(meta)
            row["umap_1"] = float(f"{coord[0]:.6f}")
            row["umap_2"] = float(f"{coord[1]:.6f}")
            reduction_rows.append(row)
        _plot_umap(reduction_rows, class_rows if class_rows else None,
                   args.output_file)

    if any(x is not None for x in (class_rows or None, feature_rows or None, reduction_rows or None)):
        write_results(class_rows or None, feature_rows or None, reduction_rows or None, args.output_file)
    else:
        print("No valid results to write.")


if __name__ == "__main__":
    main()
