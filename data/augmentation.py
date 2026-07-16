"""Data augmentation pipeline for optical flow training.

Implements:
  - Photometric augmentation  (colour jitter, grayscale, gamma)
  - Spatial augmentation      (random crop, flip, rotation, scale)
  - Occlusion augmentation    (erasing patches in the second image)
  - Flow-aware spatial transforms (correctly transform GT flow)

Design:  Each augmenter is a callable that receives a dict with keys
  ``image1``, ``image2``, ``flow``, ``valid``  (all numpy arrays, H×W×3 /
  H×W×2 / H×W bool) and returns the same dict with augmented values.

A ``FlowAugmentor`` composes them for training; a ``SparseFlowAugmentor``
handles datasets with sparse GT (KITTI).
"""
from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Photometric augmentation
# ---------------------------------------------------------------------------

class ColorJitter:
    """Apply identical colour jitter to both images, then independently."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.15 / 3.14,
        asymmetric_prob: float = 0.2,
    ) -> None:
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue
        self.asym_prob  = asymmetric_prob

    def _jitter(self, img: np.ndarray) -> np.ndarray:
        params = {
            "b": random.uniform(max(0, 1 - self.brightness), 1 + self.brightness),
            "c": random.uniform(max(0, 1 - self.contrast), 1 + self.contrast),
            "s": random.uniform(max(0, 1 - self.saturation), 1 + self.saturation),
            "h": random.uniform(-self.hue, self.hue) * 180,
        }
        return self._jitter_with_params(img, params)

    def _jitter_with_params(self, img: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        img = img.astype(np.float32)
        # brightness
        b = params["b"]
        img = img * b
        # contrast
        c = params["c"]
        mean = img.mean()
        img = (img - mean) * c + mean
        # saturation (operate in BGR)
        s = params["s"]
        gray = img.mean(axis=2, keepdims=True)
        img = img * s + gray * (1 - s)
        # hue (simple hue rotation in HSV)
        h = params["h"]
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + h) % 180
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        return np.clip(img, 0, 255)

    def __call__(self, sample: Dict) -> Dict:
        img1 = sample["image1"].copy()
        img2 = sample["image2"].copy()
        # Symmetric jitter uses the same photometric parameters for both views.
        if random.random() < self.asym_prob:
            img1 = self._jitter(img1)
            img2 = self._jitter(img2)
        else:
            shared = {
                "b": random.uniform(max(0, 1 - self.brightness), 1 + self.brightness),
                "c": random.uniform(max(0, 1 - self.contrast), 1 + self.contrast),
                "s": random.uniform(max(0, 1 - self.saturation), 1 + self.saturation),
                "h": random.uniform(-self.hue, self.hue) * 180,
            }
            img1 = self._jitter_with_params(img1, shared)
            img2 = self._jitter_with_params(img2, shared)
        sample["image1"] = img1
        sample["image2"] = img2
        return sample


class GrayScaleTransform:
    """Convert both images to grayscale with given probability."""

    def __init__(self, prob: float = 0.1) -> None:
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.prob:
            for k in ("image1", "image2"):
                gray = cv2.cvtColor(
                    sample[k].astype(np.uint8), cv2.COLOR_BGR2GRAY
                )
                sample[k] = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        return sample


# ---------------------------------------------------------------------------
# Spatial augmentation
# ---------------------------------------------------------------------------

class RandomCrop:
    def __init__(self, crop_size: Tuple[int, int]) -> None:
        self.h, self.w = crop_size

    def __call__(self, sample: Dict) -> Dict:
        H, W = sample["image1"].shape[:2]
        y0 = random.randint(0, max(0, H - self.h))
        x0 = random.randint(0, max(0, W - self.w))

        def _crop(arr: np.ndarray) -> np.ndarray:
            return arr[y0:y0 + self.h, x0:x0 + self.w]

        sample["image1"] = _crop(sample["image1"])
        sample["image2"] = _crop(sample["image2"])
        if "flow" in sample and sample["flow"] is not None:
            sample["flow"]  = _crop(sample["flow"])
        if "valid" in sample and sample["valid"] is not None:
            sample["valid"] = _crop(sample["valid"])
        if "occlusion" in sample and sample["occlusion"] is not None:
            sample["occlusion"] = _crop(sample["occlusion"])
        if "invalid" in sample and sample["invalid"] is not None:
            sample["invalid"] = _crop(sample["invalid"])
        if "synthetic_occlusion" in sample and sample["synthetic_occlusion"] is not None:
            sample["synthetic_occlusion"] = _crop(sample["synthetic_occlusion"])
        return sample


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.prob:
            sample["image1"] = np.ascontiguousarray(sample["image1"][:, ::-1])
            sample["image2"] = np.ascontiguousarray(sample["image2"][:, ::-1])
            if "flow" in sample and sample["flow"] is not None:
                flow = np.ascontiguousarray(sample["flow"][:, ::-1])
                flow[:, :, 0] *= -1          # invert horizontal component
                sample["flow"] = flow
            if "valid" in sample and sample["valid"] is not None:
                sample["valid"] = np.ascontiguousarray(sample["valid"][:, ::-1])
            if "occlusion" in sample and sample["occlusion"] is not None:
                sample["occlusion"] = np.ascontiguousarray(sample["occlusion"][:, ::-1])
            if "invalid" in sample and sample["invalid"] is not None:
                sample["invalid"] = np.ascontiguousarray(sample["invalid"][:, ::-1])
            if "synthetic_occlusion" in sample and sample["synthetic_occlusion"] is not None:
                sample["synthetic_occlusion"] = np.ascontiguousarray(
                    sample["synthetic_occlusion"][:, ::-1]
                )
        return sample


class RandomVerticalFlip:
    def __init__(self, prob: float = 0.1) -> None:
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.prob:
            sample["image1"] = np.ascontiguousarray(sample["image1"][::-1])
            sample["image2"] = np.ascontiguousarray(sample["image2"][::-1])
            if "flow" in sample and sample["flow"] is not None:
                flow = np.ascontiguousarray(sample["flow"][::-1])
                flow[:, :, 1] *= -1          # invert vertical component
                sample["flow"] = flow
            if "valid" in sample and sample["valid"] is not None:
                sample["valid"] = np.ascontiguousarray(sample["valid"][::-1])
            if "occlusion" in sample and sample["occlusion"] is not None:
                sample["occlusion"] = np.ascontiguousarray(sample["occlusion"][::-1])
            if "invalid" in sample and sample["invalid"] is not None:
                sample["invalid"] = np.ascontiguousarray(sample["invalid"][::-1])
            if "synthetic_occlusion" in sample and sample["synthetic_occlusion"] is not None:
                sample["synthetic_occlusion"] = np.ascontiguousarray(
                    sample["synthetic_occlusion"][::-1]
                )
        return sample


class RandomScaleAndCrop:
    """Scale image to a random size then crop to target resolution."""

    def __init__(
        self,
        crop_size: Tuple[int, int],
        min_scale: float = -0.2,
        max_scale: float = 0.5,
        stretch_prob: float = 0.8,
        detail_crop_prob: float = 0.0,
    ) -> None:
        self.crop_h, self.crop_w = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.stretch_prob = stretch_prob
        self.detail_crop_prob = detail_crop_prob

    @staticmethod
    def _normalise_score(score: np.ndarray) -> np.ndarray:
        finite = np.isfinite(score)
        if not finite.any():
            return np.zeros_like(score, dtype=np.float32)
        values = score[finite]
        scale = np.percentile(values, 95)
        if scale <= 1e-6:
            return np.zeros_like(score, dtype=np.float32)
        return np.clip(score / scale, 0.0, 1.0).astype(np.float32)

    def _sample_crop(self, sample: Dict, new_h: int, new_w: int) -> Tuple[int, int]:
        max_y = new_h - self.crop_h
        max_x = new_w - self.crop_w

        if self.detail_crop_prob > 0 and random.random() < self.detail_crop_prob:
            flow = sample.get("flow")
            if flow is not None:
                du_dx = cv2.Sobel(flow[..., 0], cv2.CV_32F, 1, 0, ksize=3)
                du_dy = cv2.Sobel(flow[..., 0], cv2.CV_32F, 0, 1, ksize=3)
                dv_dx = cv2.Sobel(flow[..., 1], cv2.CV_32F, 1, 0, ksize=3)
                dv_dy = cv2.Sobel(flow[..., 1], cv2.CV_32F, 0, 1, ksize=3)
                boundary = np.sqrt(
                    du_dx * du_dx + du_dy * du_dy
                    + dv_dx * dv_dx + dv_dy * dv_dy
                )
                speed = np.linalg.norm(flow, axis=2)
                score = self._normalise_score(boundary)
                score += 0.75 * (speed >= 40.0).astype(np.float32)

                occlusion = sample.get("occlusion")
                if occlusion is not None:
                    occ = occlusion.astype(np.float32)
                    occ_edge = np.abs(cv2.Laplacian(occ, cv2.CV_32F))
                    score += self._normalise_score(occ_edge)
                invalid = sample.get("invalid")
                if invalid is not None:
                    score *= 1.0 - invalid.astype(np.float32)
            else:
                img = sample["image1"]
                gray = cv2.cvtColor(
                    np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
                )
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                score = self._normalise_score(np.sqrt(gx * gx + gy * gy))

            # Select from the strongest one percent rather than always taking
            # one deterministic maximum edge.
            flat = score.reshape(-1)
            count = max(1, min(flat.size, flat.size // 100))
            candidates = np.argpartition(flat, -count)[-count:]
            selected = int(random.choice(candidates.tolist()))
            yy, xx = np.unravel_index(selected, score.shape)

            y0 = int(np.clip(yy - self.crop_h // 2, 0, max_y))
            x0 = int(np.clip(xx - self.crop_w // 2, 0, max_x))
            return y0, x0

        return random.randint(0, max_y), random.randint(0, max_x)

    def __call__(self, sample: Dict) -> Dict:
        H, W = sample["image1"].shape[:2]

        # Random log-scale
        scale = 2 ** random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if random.random() < self.stretch_prob:
            scale_x *= 2 ** random.uniform(-0.1, 0.1)
            scale_y *= 2 ** random.uniform(-0.1, 0.1)

        # Ensure output is at least as large as crop
        scale_x = max(scale_x, self.crop_w / W)
        scale_y = max(scale_y, self.crop_h / H)

        new_W = int(W * scale_x + 0.5)
        new_H = int(H * scale_y + 0.5)

        def _resize_img(img: np.ndarray) -> np.ndarray:
            return cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

        sample["image1"] = _resize_img(sample["image1"])
        sample["image2"] = _resize_img(sample["image2"])

        if "flow" in sample and sample["flow"] is not None:
            flow = cv2.resize(
                sample["flow"], (new_W, new_H), interpolation=cv2.INTER_LINEAR
            )
            flow[:, :, 0] *= scale_x
            flow[:, :, 1] *= scale_y
            sample["flow"] = flow

        if "valid" in sample and sample["valid"] is not None:
            sample["valid"] = cv2.resize(
                sample["valid"].astype(np.uint8), (new_W, new_H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        if "occlusion" in sample and sample["occlusion"] is not None:
            sample["occlusion"] = cv2.resize(
                sample["occlusion"].astype(np.uint8), (new_W, new_H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        if "invalid" in sample and sample["invalid"] is not None:
            sample["invalid"] = cv2.resize(
                sample["invalid"].astype(np.uint8), (new_W, new_H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        if "synthetic_occlusion" in sample and sample["synthetic_occlusion"] is not None:
            sample["synthetic_occlusion"] = cv2.resize(
                sample["synthetic_occlusion"].astype(np.uint8), (new_W, new_H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        # Random crop
        y0, x0 = self._sample_crop(sample, new_H, new_W)

        def _crop(arr: np.ndarray) -> np.ndarray:
            return arr[y0:y0 + self.crop_h, x0:x0 + self.crop_w]

        for k in (
            "image1", "image2", "flow", "valid", "occlusion", "invalid",
            "synthetic_occlusion",
        ):
            if k in sample and sample[k] is not None:
                sample[k] = _crop(sample[k])

        return sample


# ---------------------------------------------------------------------------
# Occlusion augmentation
# ---------------------------------------------------------------------------

class RandomErase:
    """Randomly erase rectangular patches in image2 to simulate occlusions."""

    def __init__(
        self,
        prob: float = 0.5,
        max_area_ratio: float = 0.5,
        num_patches: int = 3,
    ) -> None:
        self.prob = prob
        self.max_area = max_area_ratio
        self.num_patches = num_patches

    def __call__(self, sample: Dict) -> Dict:
        h, w = sample["image2"].shape[:2]
        erase_mask = sample.get("synthetic_occlusion")
        if erase_mask is None:
            erase_mask = np.zeros((h, w), dtype=bool)
        else:
            erase_mask = erase_mask.astype(bool).copy()
        if random.random() > self.prob:
            sample["synthetic_occlusion"] = erase_mask
            return sample
        img = sample["image2"].copy()
        H, W = img.shape[:2]
        for _ in range(self.num_patches):
            dx = random.randint(W // 8, W // 2)
            dy = random.randint(H // 8, H // 2)
            x1 = random.randint(0, W - dx)
            y1 = random.randint(0, H - dy)
            mean_col = img.mean(axis=(0, 1))
            img[y1:y1 + dy, x1:x1 + dx] = mean_col
            erase_mask[y1:y1 + dy, x1:x1 + dx] = True
        sample["image2"] = img
        sample["synthetic_occlusion"] = erase_mask
        return sample


# ---------------------------------------------------------------------------
# Composed augmentors
# ---------------------------------------------------------------------------

class FlowAugmentor:
    """Full augmentation pipeline for dense-GT datasets (Sintel, Things, Spring)."""

    def __init__(
        self,
        crop_size: Tuple[int, int],
        min_scale: float = -0.2,
        max_scale: float = 0.5,
        detail_crop_prob: float = 0.0,
    ) -> None:
        self.transforms = [
            ColorJitter(),
            GrayScaleTransform(),
            RandomScaleAndCrop(crop_size, min_scale, max_scale, detail_crop_prob=detail_crop_prob),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomErase(),
        ]

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


class SparseFlowAugmentor:
    """Augmentation for sparse-GT datasets (KITTI).  Skips scale/stretch."""

    def __init__(self, crop_size: Tuple[int, int]) -> None:
        self.transforms = [
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            RandomHorizontalFlip(),
            RandomCrop(crop_size),
            RandomErase(prob=0.3),
        ]

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample
