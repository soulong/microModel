import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from torch.utils.data import DataLoader


def _preload_samples(samples, config, load_fn):
    t0 = time.time()
    preloaded = [None] * len(samples)
    n_workers = config["data"].get("preload_workers", 8)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(load_fn, item): i for i, item in enumerate(samples)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preloading"):
            preloaded[futures[future]] = future.result()
    print(f"Preloaded {len(preloaded)} images ({time.time() - t0:.1f}s)")
    return preloaded


def _make_loaders(train_ds, val_ds, config, collate_fn=None):
    num_workers = config["data"]["num_workers"]
    if platform.system() == "Windows":
        num_workers = 0

    loader_kwargs = dict(
        batch_size=config["training"]["batch_size"],
        num_workers=num_workers,
        pin_memory=True,
    )
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = config["data"].get("prefetch_factor", 4)

    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, drop_last=False, **loader_kwargs,
    )
    return train_loader, val_loader
