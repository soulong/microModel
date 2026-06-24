"""Orchestrate the full test suite: gen data → train → infer → report."""

import os
import subprocess
import sys
import re
import shutil

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DIR = os.path.dirname(__file__)
TEST_RUNS = os.path.join(REPO, "test_runs")
PYTHON = sys.executable
MICROMODEL = [PYTHON, "-m", "microModel.cli"]

results = []


def log(msg):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def run(cmd, desc, cwd=REPO):
    log(f"{desc} …")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    ok = result.returncode == 0
    status = "PASS" if ok else "FAIL"
    results.append((desc, status))
    if not ok:
        print(result.stderr)
    else:
        lines = [l for l in result.stdout.split("\n") if l.strip()][-5:]
        for l in lines:
            print(f"  {l}")
    return ok


def latest_run_dir(suffix="best_model.pth"):
    if not os.path.isdir(TEST_RUNS):
        return None
    dirs = sorted(
        [d for d in os.listdir(TEST_RUNS) if os.path.isdir(os.path.join(TEST_RUNS, d))],
        reverse=True,
    )
    for d in dirs:
        ckpt = os.path.join(TEST_RUNS, d, suffix)
        if os.path.isfile(ckpt):
            return os.path.join(TEST_RUNS, d)
    return None


def find_checkpoint_dir(mode):
    for d in sorted(os.listdir(TEST_RUNS), reverse=True):
        run_path = os.path.join(TEST_RUNS, d)
        if not os.path.isdir(run_path):
            continue
        cfg_path = os.path.join(run_path, "config.yaml")
        if not os.path.isfile(cfg_path):
            continue
        with open(cfg_path) as f:
            content = f.read()
        if mode == "sl" and "focal_loss_gamma" in content:
            return run_path
        if mode == "ssl" and "ema_momentum_start" in content:
            return run_path
    return None


def remove(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def main():
    # Ensure clean state
    for f in ["test_infer_single.db", "test_infer_single.xlsx",
              "test_infer_dataset.db", "test_infer_dataset_ssl.db"]:
        remove(os.path.join(REPO, f))

    # ── Step 1: Generate test data ──────────────────────────────────────
    run([PYTHON, "test/create_single_testdata.py"], "Generate single testdata")

    # ── Step 2: Train SL ────────────────────────────────────────────────
    sl_config = "test/configs/test_sl.yaml"
    run(MICROMODEL + ["train", "--config", sl_config, "--mode", "sl", "--post-analysis"],
        "SL train (2 epochs, no analysis)")

    # ── Step 3: Train SSL ───────────────────────────────────────────────
    ssl_config = "test/configs/test_ssl.yaml"
    run(MICROMODEL + ["train", "--config", ssl_config, "--mode", "ssl", "--post-analysis"],
        "SSL train (2 epochs, no analysis)")

    # ── Step 4: Locate checkpoints ──────────────────────────────────────
    sl_run = find_checkpoint_dir("sl")
    ssl_run = find_checkpoint_dir("ssl")
    has_sl = sl_run is not None and os.path.isfile(os.path.join(sl_run, "best_model.pth"))
    has_ssl = ssl_run is not None and os.path.isfile(os.path.join(ssl_run, "best_model.pth"))
    results.append(("SL checkpoint found", "PASS" if has_sl else "FAIL"))
    results.append(("SSL checkpoint found", "PASS" if has_ssl else "FAIL"))

    if has_sl:
        ckpt = os.path.join(sl_run, "best_model.pth")
        # ── Step 5: SL inference (single) ───────────────────────────────
        run(MICROMODEL + ["inference",
             "--checkpoint", ckpt,
             "--input_dir", "test/testdata_single",
             "--output_type", "class",
             "--output_type", "feature",
             "--output_file", "test_infer_single.db"],
            "SL inference (single mode)")

        # ── Step 6: SL inference (xlsx) ─────────────────────────────────
        run(MICROMODEL + ["inference",
             "--checkpoint", ckpt,
             "--input_dir", "test/testdata_single",
             "--output_type", "class",
             "--output_file", "test_infer_single.xlsx"],
            "SL inference (xlsx output)")

    if has_ssl:
        ckpt = os.path.join(ssl_run, "best_model.pth")
        # ── Step 8: SSL inference (single) ──────────────────────────────
        run(MICROMODEL + ["inference",
             "--checkpoint", ckpt,
             "--input_dir", "test/testdata_single",
             "--output_type", "feature",
             "--output_file", "test_infer_single.db"],
            "SSL inference (single mode)")

    # ── Step 9: Verify output files ────────────────────────────────────
    expected = [
        ("test_runs/ (dir)", os.path.isdir(TEST_RUNS)),
        ("testdata_single/ (dir)", os.path.isdir(os.path.join(TEST_DIR, "testdata_single"))),
    ]
    if has_sl:
        expected += [
            ("SL best_model.pth", os.path.isfile(os.path.join(sl_run, "best_model.pth"))),
            ("SL config.yaml", os.path.isfile(os.path.join(sl_run, "config.yaml"))),
            ("SL training_log.jsonl", os.path.isfile(os.path.join(sl_run, "training_log.jsonl"))),
            ("test_infer_single.db", os.path.isfile(os.path.join(REPO, "test_infer_single.db"))),
            ("test_infer_single.xlsx", os.path.isfile(os.path.join(REPO, "test_infer_single.xlsx"))),
        ]
    if has_ssl:
        expected += [
            ("SSL best_model.pth", os.path.isfile(os.path.join(ssl_run, "best_model.pth"))),
            ("SSL config.yaml", os.path.isfile(os.path.join(ssl_run, "config.yaml"))),
            ("SSL training_log.jsonl", os.path.isfile(os.path.join(ssl_run, "training_log.jsonl"))),
        ]
    for name, ok in expected:
        results.append((name, "PASS" if ok else "FAIL"))

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    n_pass = sum(1 for _, s in results if s == "PASS")
    n_fail = sum(1 for _, s in results if s == "FAIL")
    for name, status in results:
        print(f"  [{status}]  {name}")
    print(f"\n  {n_pass} passed, {n_fail} failed out of {len(results)} tests")

    # ── Report -----------------------------------------------------------------
    if "FAIL" in [s for _, s in results]:
        print("\n  WARNING: some tests FAILED — inspect output above.")
    else:
        print("\n  All tests PASSED.")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
