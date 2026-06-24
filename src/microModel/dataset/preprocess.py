import numpy as np
import tifffile


def preprocess_array(img, config):
    """Preprocess an in-memory image array through the full pipeline.

    Same steps as load_and_preprocess() but starts from a numpy array
    instead of a file path. Handles (C,H,W), (H,W,C), and (H,W) layouts.

    Parameters
    ----------
    img : np.ndarray
        Raw image pixels.
    config : dict
        Config dict with ``model`` and ``strategy`` keys.

    Returns
    -------
    img : np.ndarray
        (C, target_size, target_size) float32 array, normalized.
    mask : np.ndarray or None
        (target_size, target_size) boolean mask, or None.
    """
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
    elif img.ndim == 3 and img.shape[-1] <= 16:
        img = img.transpose(2, 0, 1)

    img = select_channels(img, config["model"]["channel_indices"])

    strat = config.get("strategy", {})
    cell_mask = strat.get("cell_mask", True)
    scale_pad = strat.get("scale_pad", True)
    should_normalize = strat.get("normalize", True)

    mask = None
    if cell_mask:
        mask = img[0] > 0

    if should_normalize:
        img = normalize_image(img, mask, img.shape[0])

    img, mask = pad_and_resize(img, mask, config["model"]["target_size"], scale_pad)
    return img, mask


def select_channels(img, channel_indices):
    actual = img.shape[0]
    idx = [i - 1 for i in channel_indices]
    if max(idx) >= actual:
        raise ValueError(
            f"max channel_indices {max(channel_indices)} exceeds "
            f"actual channel count {actual} in image"
        )
    return img[idx]


def normalize_image(img, mask, n_channels):
    if mask is not None:
        for c in range(n_channels):
            fg = img[c][mask]
            if len(fg) == 0:
                continue
            if fg.std() > 1e-8:
                img[c][mask] = (fg - fg.mean()) / fg.std()
            else:
                img[c][mask] = fg - fg.mean()
    else:
        for c in range(n_channels):
            mean = img[c].mean()
            std = img[c].std()
            if std > 1e-8:
                img[c] = (img[c] - mean) / std
            else:
                img[c] = img[c] - mean

    if mask is not None:
        img[:, ~mask] = 0.0

    return img


def pad_and_resize(img, mask, target_size, scale_pad):
    C, H, W = img.shape
    t = target_size
    max_side = max(H, W)

    pad_h = max_side - H
    pad_w = max_side - W
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img = np.pad(img, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                 mode="constant", constant_values=0)
    if mask is not None:
        mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)),
                      mode="constant", constant_values=False)

    if scale_pad or max_side > t:
        from skimage.transform import resize as sk_resize
        img = sk_resize(img.transpose(1, 2, 0), (t, t),
                        order=1, preserve_range=True, anti_aliasing=True).transpose(2, 0, 1)
        if mask is not None:
            mask = sk_resize(mask.astype(np.float32), (t, t),
                             order=0, preserve_range=True).astype(bool)
    elif max_side < t:
        extra = t - max_side
        extra_top = extra // 2
        extra_bottom = extra - extra_top
        img = np.pad(img, ((0, 0), (extra_top, extra_bottom), (extra_top, extra_bottom)),
                     mode="constant", constant_values=0)
        if mask is not None:
            mask = np.pad(mask, ((extra_top, extra_bottom), (extra_top, extra_bottom)),
                          mode="constant", constant_values=False)

    if mask is not None:
        img[:, ~mask] = 0.0
    return img.astype(np.float32), mask


def load_and_preprocess(path, config):
    img = tifffile.imread(path).astype(np.float32)
    return preprocess_array(img, config)
