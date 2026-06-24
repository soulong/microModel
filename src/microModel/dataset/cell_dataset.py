import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

from microModel.utils import natural_sort_key
from ._shared import _preload_samples, _make_loaders
from .preprocess import load_and_preprocess


class CellDataset(Dataset):
    @classmethod
    def from_samples(cls, samples, classes, config, is_training=True, augmentation=None):
        ds = cls.__new__(cls)
        ds.config = config
        ds.is_training = is_training
        ds.augmentation = augmentation
        ds.classes = sorted(classes, key=natural_sort_key)
        ds.class_to_idx = {c: i for i, c in enumerate(ds.classes)}
        ds.samples = samples
        ds._is_preloaded = False
        ds._preloaded = []
        return ds

    def __init__(self, root_dirs, config, is_training=True, augmentation=None):
        self.config = config
        self.is_training = is_training
        self.augmentation = augmentation
        self._is_preloaded = False
        self._preloaded = []

        if not root_dirs:
            raise ValueError("CellDataset requires at least one root directory")

        all_classes = set()
        for d in root_dirs:
            all_classes.update(
                cls for cls in os.listdir(d)
                if os.path.isdir(os.path.join(d, cls))
            )
        self.classes = sorted(all_classes, key=natural_sort_key)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for d in root_dirs:
            for cls in os.listdir(d):
                if cls not in self.class_to_idx:
                    continue
                cls_dir = os.path.join(d, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith((".tiff", ".tif")):
                        path = os.path.join(cls_dir, fname)
                        self.samples.append((path, self.class_to_idx[cls]))

    def _preload_samples(self):
        def _load_one(path_label):
            path, label = path_label
            return load_and_preprocess(path, self.config)

        self._preloaded = _preload_samples(self.samples, self.config, _load_one)
        self._is_preloaded = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if self._is_preloaded:
            img, mask = self._preloaded[idx]
            img = img.copy()
            if mask is not None:
                mask = mask.copy()
        else:
            img, mask = load_and_preprocess(path, self.config)

        if self.is_training and self.augmentation is not None:
            if mask is not None and not mask.any():
                mask = None
            img, mask = self.augmentation(img, mask)

        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

    def get_labels(self):
        return [label for _, label in self.samples]


class CellAugmentation:
    def __init__(self, config):
        aug = config["augmentation"]
        self.rotation = aug["rotation_degrees"]
        self.hflip_prob = aug["hflip_prob"]
        self.vflip_prob = aug["vflip_prob"]
        self.scale_range = aug["scale_range"]
        self.translate_range = aug["translate_range"]
        self.brightness = aug.get("brightness", 0.0)
        self.contrast = aug.get("contrast", 0.0)
        self.apply_prob = aug.get("apply_prob", 0.9)

    def __call__(self, image, mask):
        if random.random() > self.apply_prob:
            return image, mask

        wants_mask = mask is not None
        img_t = torch.from_numpy(image).float()

        angle = random.uniform(-self.rotation, self.rotation)
        img_t = TF.rotate(img_t, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)

        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        tx = random.uniform(-self.translate_range[0], self.translate_range[0]) * image.shape[2]
        ty = random.uniform(-self.translate_range[1], self.translate_range[1]) * image.shape[1]
        img_t = TF.affine(img_t, angle=0, translate=(tx, ty), scale=scale, shear=0,
                          interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)

        if random.random() < self.hflip_prob:
            img_t = TF.hflip(img_t)
        if random.random() < self.vflip_prob:
            img_t = TF.vflip(img_t)

        if wants_mask:
            mask = (img_t[0] > 0).cpu().numpy()

        img = img_t.numpy()

        if wants_mask:
            img[:, ~mask] = 0.0

        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            if wants_mask:
                img[:, mask] = img[:, mask] * factor
            else:
                img = img * factor

        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            for c in range(img.shape[0]):
                if wants_mask:
                    if not mask.any():
                        continue
                    fg_mean = img[c, mask].mean()
                    img[c, mask] = (img[c, mask] - fg_mean) * factor + fg_mean
                else:
                    mean = img[c].mean()
                    img[c] = (img[c] - mean) * factor + mean

        if wants_mask:
            img[:, ~mask] = 0.0
        return img, mask


def _stratified_sample_indices(labels, n_total):
    classes = sorted(set(labels))
    n_per_class = max(1, n_total // len(classes))

    sampled = []
    for c in classes:
        idx = [i for i, lbl in enumerate(labels) if lbl == c]
        if len(idx) > n_per_class:
            idx = list(np.random.choice(idx, n_per_class, replace=False))
        sampled.extend(idx)

    if len(sampled) > n_total:
        sampled = list(np.random.choice(sampled, n_total, replace=False))
    return sampled



def create_dataloaders_from_dirs(config):
    raw = config["data"].get("train_dir")
    if raw is None:
        raise ValueError("data.train_dir must be set in config")
    if isinstance(raw, str):
        raise ValueError(
            "data.train_dir must be a list of directories. "
            "Change from single string to list, e.g.:\n"
            "  train_dir:\n    - \"path/to/data\""
        )
    train_dir = list(raw)
    val_dir = config["data"].get("val_dir")
    sample_n = config["data"].get("sample_n")
    sample_by = config["data"].get("sample_by", "dataset")

    def _sample_per_dir(dirs, n_per):
        all_classes = set()
        for d in dirs:
            all_classes.update(
                cls for cls in os.listdir(d)
                if os.path.isdir(os.path.join(d, cls))
            )
        classes_sorted = sorted(all_classes, key=natural_sort_key)
        class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
        merged = []
        for d in dirs:
            dir_samples = []
            for cls in os.listdir(d):
                if cls not in class_to_idx:
                    continue
                cls_dir = os.path.join(d, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith((".tiff", ".tif")):
                        dir_samples.append((os.path.join(cls_dir, fname), class_to_idx[cls]))
            labels = [lbl for _, lbl in dir_samples]
            idx = _stratified_sample_indices(labels, n_per)
            for i in idx:
                merged.append(dir_samples[i])
        return merged, classes_sorted

    if val_dir is None:
        if sample_n is not None and sample_by == "dataset":
            samples, classes = _sample_per_dir(train_dir, sample_n)
            full_dataset = CellDataset.from_samples(samples, classes, config)
            train_dataset = CellDataset.from_samples(
                samples, classes, config, is_training=True,
                augmentation=CellAugmentation(config)
            )
            val_dataset = CellDataset.from_samples(samples, classes, config)
        else:
            full_dataset = CellDataset(train_dir, config, is_training=False, augmentation=None)
            train_dataset = CellDataset(train_dir, config, is_training=True, augmentation=CellAugmentation(config))
            val_dataset = CellDataset(train_dir, config, is_training=False, augmentation=None)

        all_indices = list(range(len(full_dataset)))
        labels = full_dataset.get_labels()

        if sample_n is not None and sample_by == "ensemble":
            all_indices = _stratified_sample_indices(labels, sample_n)
            labels = [labels[i] for i in all_indices]

        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=1.0 - config["data"]["train_val_split"],
            stratify=labels
        )
        if not (sample_n is not None and sample_by == "dataset"):
            train_dataset = CellDataset(train_dir, config, is_training=True, augmentation=CellAugmentation(config))
            val_dataset = CellDataset(train_dir, config, is_training=False, augmentation=None)
        if config["data"].get("preload"):
            full_dataset._preload_samples()
            train_dataset._preloaded = full_dataset._preloaded
            train_dataset._is_preloaded = True
            val_dataset._preloaded = full_dataset._preloaded
            val_dataset._is_preloaded = True
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)
        label_map = full_dataset.class_to_idx
    else:
        if sample_n is not None and sample_by == "dataset":
            train_samples, train_classes = _sample_per_dir(train_dir, sample_n)
            train_dataset = CellDataset.from_samples(train_samples, train_classes, config, is_training=True, augmentation=CellAugmentation(config))
            val_dirs = [val_dir]
            val_samples, val_classes = _sample_per_dir(val_dirs, sample_n)
            val_dataset = CellDataset.from_samples(val_samples, val_classes, config, is_training=False)
            if config["data"].get("preload"):
                train_dataset._preload_samples()
                val_dataset._preload_samples()
        else:
            train_dataset = CellDataset(train_dir, config, is_training=True, augmentation=CellAugmentation(config))
            val_dirs = [val_dir]
            val_dataset = CellDataset(val_dirs, config, is_training=False, augmentation=None)
            if config["data"].get("preload"):
                train_dataset._preload_samples()
                val_dataset._preload_samples()
            if sample_n is not None:
                ratio = config["data"]["train_val_split"]
                n_train = int(sample_n * ratio)
                n_val = sample_n - n_train
                train_labels = train_dataset.get_labels()
                val_labels = val_dataset.get_labels()
                train_idx = _stratified_sample_indices(train_labels, n_train)
                val_idx = _stratified_sample_indices(val_labels, n_val)
                train_dataset = Subset(train_dataset, train_idx)
                val_dataset = Subset(val_dataset, val_idx)

        train_subset = train_dataset
        val_subset = val_dataset
        label_map = train_dataset.class_to_idx

    train_loader, val_loader = _make_loaders(train_subset, val_subset, config)
    return train_loader, val_loader, label_map
