"""CLI entry point for microModel.

Usage
-----
  micromodel train --config <yaml> [--mode {sl,ssl}]
  micromodel inference --checkpoint <pth> [options...]

Also callable from root shims (train_sl.py, train_ssl_dinov3.py,
inference.py) which prepend their respective subcommand.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="micromodel",
        description="Microscopy image classification and feature extraction",
    )
    sub = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # ── train ──────────────────────────────────────────────────────────
    t = sub.add_parser("train", help="Train a model (SL or SSL)")
    t.add_argument("--config", required=True, help="Path to YAML config file")
    t.add_argument("--mode", choices=["sl", "ssl"], default="sl",
                   help="Training mode: sl (supervised, default) or ssl (self-supervised)")
    t.add_argument("--post-analysis", action="store_true",
                   help="Run post-training analysis (UMAP, embeddings, predictions)")

    # ── inference ──────────────────────────────────────────────────────
    i = sub.add_parser("inference", help="Run inference on images or a microProfiler dataset")
    i.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    i.add_argument("--input_dir", default=None, help="Directory of TIFF images")
    i.add_argument("--filelist", default=None,
                   help="File with one image path per line")
    i.add_argument("--output_file", default="results.db",
                   help="Output path — .db or .xlsx (default: results.db)")
    i.add_argument("--output_type", action="append",
                   choices=["class", "feature", "reduction"],
                   help="Output type(s) to produce (repeatable)")
    i.add_argument("--reducer", default=None,
                   help="Fitted UMAP reducer .pkl (required if reduction in --output_type)")
    i.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Device (default: auto)")
    i.add_argument("--checkpoint_type", choices=["auto", "sl", "ssl"], default="auto",
                   help="Checkpoint type (default: auto-detect)")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str or None
        Arguments to parse. If ``None``, ``sys.argv[1:]`` is used.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        _dispatch_train(args)
    elif args.command == "inference":
        _dispatch_inference(args)


def _dispatch_train(args: argparse.Namespace) -> None:
    if args.mode == "sl":
        from microModel.train_sl import train
        train(config_path=args.config, post_analysis=args.post_analysis)
    else:
        from microModel.train_ssl_dinov3 import train
        train(config_path=args.config, post_analysis=args.post_analysis)


def _dispatch_inference(args: argparse.Namespace) -> None:
    from microModel.inference import main as inference_main
    args.device = getattr(args, "device", "auto")
    inference_main(args)


if __name__ == "__main__":
    main()
