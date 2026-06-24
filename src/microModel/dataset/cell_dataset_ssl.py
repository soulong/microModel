import os
import numpy as np
import torch
from torch.utils.data import Dataset

from ._shared import _preload_samples, _make_loaders
from .preprocess import load_and_preprocess
from .augmentation_ssl import MultiCropAugmentation


class CellDatasetSSL(Dataset):
    def __init__(self, root_dirs, config):
        self.config = config
        self._is_preloaded = False
        self._preloaded = []

        if not root_dirs:
            raise ValueError("CellDatasetSSL requires at least one root directory")

        self.samples = []
        for d in root_dirs:
            for root, _, files in os.walk(d):
                for fname in sorted(files):
                    if fname.lower().endswith((".tiff", ".tif")):
                        self.samples.append(os.path.join(root, fname))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No .tiff files found in {root_dirs}")

    @classmethod
    def from_file_list(cls, file_list, config):
        ds = cls.__new__(cls)
        ds.config = config
        ds._is_preloaded = False
        ds._preloaded = []
        ds.samples = file_list
        return ds

    def _preload_samples(self):
        def _load_one(path):
            img, mask = load_and_preprocess(path, self.config)
            img_tensor = torch.from_numpy(img).float()
            if mask is not None:
                mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
            else:
                mask_tensor = torch.ones_like(img_tensor[0:1])
            return {"image": img_tensor, "mask": mask_tensor}

        self._preloaded = _preload_samples(self.samples, self.config, _load_one)
        self._is_preloaded = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._is_preloaded:
            item = self._preloaded[idx]
            return {"image": item["image"].clone(), "mask": item["mask"].clone()}

        path = self.samples[idx]
        img, mask = load_and_preprocess(path, self.config)
        img_tensor = torch.from_numpy(img).float()
        if mask is not None:
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        else:
            mask_tensor = torch.ones_like(img_tensor[0:1])
        return {"image": img_tensor, "mask": mask_tensor}


def _sample_per_dir_ssl(dirs, n_per):
    merged = []
    for d in dirs:
        files = []
        for root, _, fnames in os.walk(d):
            for fname in sorted(fnames):
                if fname.lower().endswith((".tiff", ".tif")):
                    files.append(os.path.join(root, fname))
        if n_per is not None and len(files) > n_per:
            files = list(np.random.choice(files, n_per, replace=False))
        merged.extend(files)
    return merged


class _CollateWithAug:
    def __init__(self, augmentation):
        self.augmentation = augmentation
    def __call__(self, batch):
        return _collate_with_aug(batch, self.augmentation)

def create_ssl_dataloader(config):
    train_dir = config["data"].get("train_dir")
    if isinstance(train_dir, str):
        train_dir = [train_dir]
    val_dir = config["data"].get("val_dir")
    sample_n = config["data"].get("sample_n")
    sample_by = config["data"].get("sample_by", "dataset")
    augmentation = MultiCropAugmentation.from_config(config)



    if val_dir is None:
        if sample_n is not None and sample_by == "dataset":
            samples = _sample_per_dir_ssl(train_dir, sample_n)
            full_dataset = CellDatasetSSL.from_file_list(samples, config)
        else:
            full_dataset = CellDatasetSSL(train_dir, config)

        if config["data"].get("preload"):
            full_dataset._preload_samples()

        all_indices = list(range(len(full_dataset)))

        if sample_n is not None and sample_by == "ensemble":
            n_total = min(sample_n, len(all_indices))
            all_indices = sorted(np.random.choice(all_indices, n_total, replace=False))

        split = config["data"].get("train_val_split", 0.8)
        np.random.shuffle(all_indices)
        n_train = int(len(all_indices) * split)
        train_idx = all_indices[:n_train]
        val_idx = all_indices[n_train:]

        train_ds = torch.utils.data.Subset(full_dataset, train_idx)
        val_ds = torch.utils.data.Subset(full_dataset, val_idx)
    else:
        if isinstance(val_dir, str):
            val_dir = [val_dir]

        if sample_n is not None and sample_by == "dataset":
            train_samples = _sample_per_dir_ssl(train_dir, sample_n)
            train_ds = CellDatasetSSL.from_file_list(train_samples, config)
            val_samples = _sample_per_dir_ssl(val_dir, sample_n)
            val_ds = CellDatasetSSL.from_file_list(val_samples, config)
            if config["data"].get("preload"):
                train_ds._preload_samples()
                val_ds._preloaded = train_ds._preloaded
                val_ds._is_preloaded = True
        else:
            train_ds = CellDatasetSSL(train_dir, config)
            val_ds = CellDatasetSSL(val_dir, config)
            if config["data"].get("preload"):
                train_ds._preload_samples()
                val_ds._preloaded = train_ds._preloaded
                val_ds._is_preloaded = True
            if sample_n is not None:
                ratio = config["data"]["train_val_split"]
                n_train = max(1, int(sample_n * ratio))
                n_val = max(1, sample_n - n_train)
                train_idx = sorted(np.random.choice(range(len(train_ds)), min(n_train, len(train_ds)), replace=False))
                val_idx = sorted(np.random.choice(range(len(val_ds)), min(n_val, len(val_ds)), replace=False))
                train_ds = torch.utils.data.Subset(train_ds, train_idx)
                val_ds = torch.utils.data.Subset(val_ds, val_idx)

    train_loader, val_loader = _make_loaders(
        train_ds, val_ds, config, collate_fn=_CollateWithAug(augmentation),
    )
    return train_loader, val_loader


def _collate_with_aug(batch_items, augmentation):
    views_list = []
    masks_list = []
    for item in batch_items:
        views, masks = augmentation(item["image"], item["mask"])
        views_list.append(views)
        masks_list.append(masks)

    num_crops = len(views_list[0])
    views_out = []
    masks_out = []
    for i in range(num_crops):
        views_out.append(torch.stack([v[i] for v in views_list], dim=0))
        masks_out.append(torch.stack([m[i] for m in masks_list], dim=0))

    return {"views": views_out, "masks": masks_out}
