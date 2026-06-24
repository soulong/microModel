import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms.functional as VF
from torchvision.transforms.v2 import Transform


class Stage1Spatial(Transform):
    def __init__(self, rotation_degrees: float = 360.0, flip_prob: float = 0.5,
                 elastic_alpha: float = 50.0, elastic_sigma: float = 5.0,
                 use_elastic: bool = False):
        super().__init__()
        self.use_elastic = use_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.rotation = rotation_degrees
        self.flip_prob = flip_prob

    def _elastic(self, image, mask, generator):
        B, C, H, W = image.shape
        device = image.device

        dx = torch.rand(1, H, W, generator=generator, device=device) * 2 - 1
        dy = torch.rand(1, H, W, generator=generator, device=device) * 2 - 1
        dx = F.gaussian_blur(dx, kernel_size=(21, 21), sigma=self.elastic_sigma)
        dy = F.gaussian_blur(dy, kernel_size=(21, 21), sigma=self.elastic_sigma)
        dx = dx * self.elastic_alpha
        dy = dy * self.elastic_alpha

        x, y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
        x = x.float().unsqueeze(0) + dx
        y = y.float().unsqueeze(0) + dy
        x = 2 * x / (W - 1) - 1
        y = 2 * y / (H - 1) - 1
        grid = torch.stack([x, y], dim=-1).squeeze(0)

        img_warped = F.grid_sample(image, grid.expand(B, -1, -1, -1), mode='bilinear', align_corners=False)
        if mask is not None:
            msk_warped = F.grid_sample(mask.float(), grid.expand(B, -1, -1, -1), mode='nearest', align_corners=False)
            return img_warped, msk_warped
        return img_warped, None

    def forward(self, image: torch.Tensor, mask: torch.Tensor,
                generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0).unsqueeze(0) if mask is not None else None

        if self.use_elastic:
            image, mask = self._elastic(image, mask, generator)

        angle = random.uniform(-self.rotation, self.rotation)
        image = VF.rotate(image, angle, interpolation=VF.InterpolationMode.BILINEAR, fill=0.0)
        if mask is not None:
            mask = VF.rotate(mask, angle, interpolation=VF.InterpolationMode.NEAREST, fill=0.0)

        if random.random() < self.flip_prob:
            image = VF.hflip(image)
            if mask is not None:
                mask = VF.hflip(mask)

        return image.squeeze(0), mask.squeeze(0).squeeze(0) if mask is not None else mask


class ChannelDropout(Transform):
    def __init__(self, dropout_prob: Tuple[float, float] = (0.15, 0.25)):
        super().__init__()
        self.dropout_prob = dropout_prob

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        C = image.shape[-3] if image.dim() >= 3 else 1
        for c in range(C):
            if random.random() < random.uniform(*self.dropout_prob):
                image[..., c, :, :] = 0.0
        return image


class Stage2Intensity(Transform):
    def __init__(self, noise_std_range=(0.05, 0.15), blur_sigma_range=(0.1, 1.0),
                 brightness_range=(0.85, 1.15), channel_dropout_prob=(0.15, 0.25)):
        super().__init__()
        self.noise_std_range = noise_std_range
        self.blur_sigma_range = blur_sigma_range
        self.brightness_range = brightness_range
        self.channel_dropout = ChannelDropout(dropout_prob=channel_dropout_prob)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.unsqueeze(0)

        if random.random() < 0.8:
            noise_std = random.uniform(*self.noise_std_range)
            image = image + torch.randn_like(image) * noise_std

        if random.random() < 0.2:
            k = random.choice([3, 5, 7])
            blur_sigma = random.uniform(*self.blur_sigma_range)
            ks = int(blur_sigma * 6) | 1
            ks = min(max(ks, 3), 15)
            image = VF.gaussian_blur(image, kernel_size=(ks, ks), sigma=blur_sigma)

        if random.random() < 0.5:
            factor = random.uniform(*self.brightness_range)
            image = image * factor
            image = image.clamp(-10.0, 10.0)

        return self.channel_dropout(image.squeeze(0))


def sample_foreground_crop(mask: torch.Tensor, scale_min: float, scale_max: float,
                           min_size: int) -> Optional[Tuple[int, int, int, int]]:
    fg_indices = torch.nonzero(mask)
    if fg_indices.size(0) < 10:
        return None

    idx = random.randint(0, fg_indices.size(0) - 1)
    cy, cx = fg_indices[idx].tolist()
    H, W = mask.shape

    scale = random.uniform(scale_min, scale_max)
    crop_size = int(math.sqrt(H * W) * scale)
    crop_size = max(crop_size, min_size)

    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - crop_size // 2)
    x2 = min(W, x1 + crop_size)
    y2 = min(H, y1 + crop_size)
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None

    return x1, y1, x2 - x1, y2 - y1


class MultiCropAugmentation:
    def __init__(self, input_size: int, num_global_crops: int = 2,
                 num_structural_crops: int = 4, num_texture_crops: int = 4,
                 structural_crop_resize: int = 96, texture_crop_resize: int = 64,
                 rotation_degrees: float = 360.0, flip_prob: float = 0.5,
                 use_elastic: bool = False, elastic_alpha: float = 50.0,
                 elastic_sigma: float = 5.0, noise_std_range=(0.05, 0.15),
                 blur_sigma_range=(0.1, 1.0), brightness_range=(0.85, 1.15),
                 channel_dropout_prob=(0.15, 0.25)):
        self.num_global_crops = num_global_crops
        self.num_structural_crops = num_structural_crops
        self.num_texture_crops = num_texture_crops
        self.structural_crop_resize = structural_crop_resize
        self.texture_crop_resize = texture_crop_resize

        self.stage1 = Stage1Spatial(
            rotation_degrees=rotation_degrees,
            flip_prob=flip_prob,
            elastic_alpha=elastic_alpha,
            elastic_sigma=elastic_sigma,
            use_elastic=use_elastic,
        )
        self.stage2 = Stage2Intensity(
            noise_std_range=noise_std_range,
            blur_sigma_range=blur_sigma_range,
            brightness_range=brightness_range,
            channel_dropout_prob=channel_dropout_prob,
        )
        self.resize_global = transforms_v2.Resize((input_size, input_size))
        self.resize_structural = transforms_v2.Resize(
            (structural_crop_resize, structural_crop_resize)
        )
        self.resize_texture = transforms_v2.Resize(
            (texture_crop_resize, texture_crop_resize)
        )

    @classmethod
    def from_config(cls, config):
        ssl_cfg = config["augmentation"]
        return cls(
            input_size=config["model"]["target_size"],
            num_global_crops=ssl_cfg.get("num_global_crops", 2),
            num_structural_crops=ssl_cfg.get("num_structural_crops", 4),
            num_texture_crops=ssl_cfg.get("num_texture_crops", 4),
            structural_crop_resize=ssl_cfg.get("structural_crop_resize", 96),
            texture_crop_resize=ssl_cfg.get("texture_crop_resize", 64),
            rotation_degrees=ssl_cfg.get("rotation_degrees", 360.0),
            flip_prob=ssl_cfg.get("flip_prob", 0.5),
            use_elastic=ssl_cfg.get("use_elastic", False),
            elastic_alpha=ssl_cfg.get("elastic_alpha", 50.0),
            elastic_sigma=ssl_cfg.get("elastic_sigma", 5.0),
            noise_std_range=ssl_cfg.get("noise_std_range", (0.05, 0.15)),
            blur_sigma_range=ssl_cfg.get("blur_sigma_range", (0.1, 1.0)),
            brightness_range=ssl_cfg.get("brightness_range", (0.85, 1.15)),
            channel_dropout_prob=ssl_cfg.get("channel_dropout_prob", (0.15, 0.25)),
        )

    def _sample_local_crop(self, mask: torch.Tensor, scale_min: float, scale_max: float,
                           min_size: int) -> Optional[Tuple[int, int, int, int]]:
        if mask.dim() == 3:
            mask = mask[0]
        return sample_foreground_crop(mask, scale_min, scale_max, min_size)

    def _apply_stage1_to_view(self, image: torch.Tensor, mask: torch.Tensor,
                                generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        ref_mask = mask[0] if mask.dim() == 3 else mask
        out_image, out_mask = self.stage1(image, ref_mask, generator=generator)
        out_mask = (out_mask > 0.5).float()
        return out_image, out_mask.squeeze(0)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        views = []
        masks = []

        spatial_seed = torch.randint(0, 2**31, (1,)).item()

        for _ in range(self.num_global_crops):
            gen = torch.Generator()
            gen.manual_seed(spatial_seed)
            img_view, msk_view = self._apply_stage1_to_view(image, mask, generator=gen)
            img_view = self.stage2(img_view)
            img_view = self.resize_global(img_view)
            msk_view = self.resize_global(msk_view.unsqueeze(0)).squeeze(0)
            msk_view = (msk_view > 0.5).float()
            views.append(img_view)
            masks.append(msk_view)

        for _ in range(self.num_structural_crops):
            gen = torch.Generator()
            gen.manual_seed(spatial_seed)
            crop = self._sample_local_crop(mask, 0.2, 0.5, 0)
            if crop is not None:
                cx, cy, crop_w, crop_h = crop
                img_patch = image[:, cy:cy + crop_h, cx:cx + crop_w]
                msk_patch = mask[:, cy:cy + crop_h, cx:cx + crop_w]
            else:
                img_patch = image.clone()
                msk_patch = mask.clone()
            img_view, msk_view = self._apply_stage1_to_view(img_patch, msk_patch, generator=gen)
            img_view = self.resize_structural(img_view)
            msk_view = self.resize_structural(msk_view.unsqueeze(0)).squeeze(0)
            msk_view = (msk_view > 0.5).float()
            views.append(img_view)
            masks.append(msk_view)

        for _ in range(self.num_texture_crops):
            gen = torch.Generator()
            gen.manual_seed(spatial_seed)
            crop = self._sample_local_crop(mask, 0.05, 0.2, 24)
            if crop is not None:
                cx, cy, crop_w, crop_h = crop
                img_patch = image[:, cy:cy + crop_h, cx:cx + crop_w]
                msk_patch = mask[:, cy:cy + crop_h, cx:cx + crop_w]
            else:
                img_patch = image.clone()
                msk_patch = mask.clone()
            img_view, msk_view = self._apply_stage1_to_view(img_patch, msk_patch, generator=gen)
            img_view = self.resize_texture(img_view)
            msk_view = self.resize_texture(msk_view.unsqueeze(0)).squeeze(0)
            msk_view = (msk_view > 0.5).float()
            views.append(img_view)
            masks.append(msk_view)

        return views, masks
