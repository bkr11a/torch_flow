"""Evaluation metrics for optical flow.

Standard metrics
----------------
epe      – Mean End-Point Error (L2 distance, averaged over valid pixels).
f1       – Outlier rate: % pixels with EPE > 3 px AND EPE > 5 % of GT magnitude
           (KITTI-2015 primary metric).

Speed-stratified EPE  (Sintel / Spring)
----------------------------------------
s0_10    – EPE on pixels with GT magnitude  0–10 px  (slow motion).
s10_40   – EPE on pixels with GT magnitude 10–40 px  (medium motion).
s40_plus – EPE on pixels with GT magnitude  > 40 px  (fast motion).

Distance-to-occlusion-boundary EPE  (Sintel, requires occlusion mask)
-----------------------------------------------------------------------
d0       – EPE on *occluded* pixels (unmatched region); requires occ_mask.
d0_10    – EPE on matched pixels within   0–10 px of nearest occlusion boundary.
d10_60   – EPE on matched pixels within  10–60 px of nearest occlusion boundary.
d60_140  – EPE on matched pixels within  60–140 px of nearest occlusion boundary.
d140_plus– EPE on matched pixels beyond 140 px from nearest occlusion boundary.

The distance buckets follow the official Sintel benchmark definition.
They reveal how well the model handles the neighbourhood of motion boundaries —
pixels near occlusion edges are harder due to ambiguous correspondences.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Distance-to-occlusion-boundary helper
# ---------------------------------------------------------------------------

def _occ_boundary_distance(occ_mask: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance transform: for every *non-occluded* pixel,
    return its distance (in pixels) to the nearest occluded pixel.

    Args:
        occ_mask: (H, W) bool array — True where the pixel is **occluded**.

    Returns:
        (H, W) float32 array.  Occluded pixels get distance 0.
        Non-occluded pixels get the distance to the nearest occluded pixel.
        If there are no occluded pixels at all, every pixel gets inf.
    """
    from scipy.ndimage import distance_transform_edt
    # distance_transform_edt measures distance to the nearest *zero* pixel.
    # Invert: occluded (True) → 0, non-occluded (False) → distance to nearest 0.
    dist = distance_transform_edt(~occ_mask).astype(np.float32)
    # Occluded pixels themselves are defined as distance 0.
    dist[occ_mask] = 0.0
    return dist


# ---------------------------------------------------------------------------
# Main metrics function
# ---------------------------------------------------------------------------

def compute_metrics(
    pred:     torch.Tensor,                    # (2, H, W) or (B, 2, H, W)
    gt:       torch.Tensor,                    # (2, H, W) or (B, 2, H, W)
    valid:    Optional[torch.Tensor] = None,   # (H, W) or (B, H, W), bool/float
    occ_mask: Optional[torch.Tensor] = None,   # (H, W) or (B, H, W), bool/float
    invalid_mask: Optional[torch.Tensor] = None,  # (H, W) or (B, H, W)
                                               # True = occluded pixel
) -> Dict[str, float]:
    """
    Compute standard optical flow evaluation metrics.

    Args:
        pred:     Predicted flow tensor.
        gt:       Ground-truth flow tensor.
        valid:    Boolean mask of pixels with reliable GT.  If the Sintel
                  dataloader is used with use_occlusions=True, occluded pixels
                  are already zeroed out here.  Pass the raw valid mask
                  (without occlusion applied) when you also supply occ_mask,
                  so that d0 (occluded EPE) can be computed separately.
        occ_mask: Optional occlusion mask (True = occluded/unmatched).
                  Required for the d0 / d0_10 / d10_60 / d60_140 / d140_plus
                  metrics.  When absent those keys are omitted from the output.

    Returns:
        Dict with keys: epe, f1, s0_10, s10_40, s40_plus,
        and optionally:  d0, d0_10, d10_60, d60_140, d140_plus.
    """
    # ---- Normalise to (B, ...) -----------------------------------------
    if pred.dim() == 3:
        pred  = pred.unsqueeze(0)
        gt    = gt.unsqueeze(0)
        if valid    is not None: valid    = valid.unsqueeze(0)
        if occ_mask is not None: occ_mask = occ_mask.unsqueeze(0)
        if invalid_mask is not None: invalid_mask = invalid_mask.unsqueeze(0)

    epe_map = (pred - gt).pow(2).sum(dim=1).sqrt()   # (B, H, W)
    mag     = gt.pow(2).sum(dim=1).sqrt()             # (B, H, W)

    if valid is None:
        valid = torch.ones_like(epe_map, dtype=torch.bool)
    else:
        valid = valid.bool()

    if occ_mask is not None:
        occ_mask = occ_mask.bool()

    if invalid_mask is not None:
        invalid_mask = invalid_mask.bool()

    reliable = torch.isfinite(gt[:, 0]) & torch.isfinite(gt[:, 1])
    if invalid_mask is not None:
        reliable = reliable & (~invalid_mask)

    matched_mask = valid & reliable
    if occ_mask is not None:
        matched_mask = matched_mask & (~occ_mask)

    if occ_mask is not None:
        unmatched_mask = occ_mask & reliable
    else:
        unmatched_mask = torch.zeros_like(matched_mask)

    all_mask = matched_mask | unmatched_mask

    epe_matched = epe_map[matched_mask]
    epe_unmatched = epe_map[unmatched_mask]
    epe_all = epe_map[all_mask]

    # ---- Standard metrics on valid (matched) pixels --------------------
    epe_valid = epe_matched
    mag_valid = mag[matched_mask]

    out: Dict[str, float] = {}
    out["epe_matched"] = epe_matched.mean().item() if epe_matched.numel() > 0 else float("nan")
    out["epe_unmatched"] = epe_unmatched.mean().item() if epe_unmatched.numel() > 0 else float("nan")
    out["epe_all"] = epe_all.mean().item() if epe_all.numel() > 0 else float("nan")
    # Backward compatibility for existing trainer/logger code.
    out["epe"] = out["epe_matched"]

    if epe_valid.numel() == 0:
        base_keys = (
            "epe", "epe_matched", "epe_unmatched", "epe_all",
            "f1", "s0_10", "s10_40", "s40_plus"
        )
        return {k: float("nan") for k in base_keys}

    # F1-all (KITTI)
    bad = (epe_valid > 3.0) & (epe_valid / mag_valid.clamp_min(1e-4) > 0.05)
    out["f1"] = bad.float().mean().item() * 100.0

    # Speed-stratified EPE
    for key, lo, hi in (
        ("s0_10",    0,   10),
        ("s10_40",  10,   40),
        ("s40_plus", 40, 1e9),
    ):
        mask = (mag_valid >= lo) & (mag_valid < hi)
        out[key] = epe_valid[mask].mean().item() if mask.any() else float("nan")

    # Spatial high-frequency recovery and phase/location alignment.  Reporting
    # both prevents an increase in boundary noise from being mistaken for
    # successful detail recovery.
    def high_pass(value: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(
            value,
            kernel_size=2,
            stride=2,
            ceil_mode=True,
            count_include_pad=False,
        )
        reconstruction = low.repeat_interleave(2, dim=-2).repeat_interleave(
            2, dim=-1
        )
        reconstruction = reconstruction[
            ..., : value.shape[-2], : value.shape[-1]
        ]
        return value - reconstruction

    pred_high = high_pass(pred)
    gt_high = high_pass(gt)
    high_mask = all_mask.unsqueeze(1).to(pred.dtype)
    numerator = (pred_high.abs() * high_mask).sum()
    denominator = (gt_high.abs() * high_mask).sum().clamp_min(1e-6)
    out["hf_recovery"] = (numerator / denominator).item()
    pred_vector = (pred_high * high_mask).flatten(1)
    gt_vector = (gt_high * high_mask).flatten(1)
    alignment = (pred_vector * gt_vector).sum(1) / (
        pred_vector.norm(dim=1) * gt_vector.norm(dim=1)
    ).clamp_min(1e-6)
    out["hf_alignment"] = alignment.mean().item()

    gt_dx = F.pad(gt[..., 1:] - gt[..., :-1], (0, 1, 0, 0))
    gt_dy = F.pad(gt[..., 1:, :] - gt[..., :-1, :], (0, 0, 0, 1))
    boundary_strength = torch.sqrt(
        gt_dx.square().sum(1) + gt_dy.square().sum(1) + 1e-9
    )
    boundary_mask = (boundary_strength > 1.0) & all_mask
    out["boundary_epe"] = (
        epe_map[boundary_mask].mean().item()
        if boundary_mask.any()
        else float("nan")
    )

    # ---- Distance-to-occlusion-boundary metrics (Sintel) ---------------
    if occ_mask is not None:
        occ_np  = occ_mask.bool().cpu().numpy()    # (B, H, W)
        epe_np  = epe_map.cpu().numpy()            # (B, H, W)
        valid_np = valid.cpu().numpy()             # (B, H, W)

        d0_epes      = []
        d0_10_epes   = []
        d10_60_epes  = []
        d60_140_epes = []
        d140_epes    = []

        for b in range(occ_np.shape[0]):
            occ_b   = occ_np[b]    # (H, W) bool
            epe_b   = epe_np[b]    # (H, W)
            valid_b = valid_np[b]  # (H, W)

            # d0: EPE on occluded pixels (no valid requirement — these are
            # intentionally excluded from valid, but we evaluate them here
            # specifically to measure quality in the unmatched region)
            occ_pixels = occ_b
            if occ_pixels.any():
                d0_epes.append(epe_b[occ_pixels].mean())

            # Distance transform on non-occluded matched pixels
            dist = _occ_boundary_distance(occ_b)   # (H, W)
            matched = valid_b & ~occ_b             # non-occluded + valid GT

            if not matched.any():
                continue

            dist_matched = dist[matched]
            epe_matched  = epe_b[matched]

            for bucket_epes, lo, hi in (
                (d0_10_epes,   0,   10),
                (d10_60_epes,  10,  60),
                (d60_140_epes, 60, 140),
                (d140_epes,   140, 1e9),
            ):
                sel = (dist_matched >= lo) & (dist_matched < hi)
                if sel.any():
                    bucket_epes.append(epe_matched[sel].mean())

        def _mean_or_nan(lst):
            return float(np.mean(lst)) if lst else float("nan")

        out["d0"]       = _mean_or_nan(d0_epes)
        out["d0_10"]    = _mean_or_nan(d0_10_epes)
        out["d10_60"]   = _mean_or_nan(d10_60_epes)
        out["d60_140"]  = _mean_or_nan(d60_140_epes)
        out["d140_plus"]= _mean_or_nan(d140_epes)

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(results_list: list) -> Dict[str, float]:
    """Average a list of per-sample metric dicts (excluding NaN entries)."""
    if not results_list:
        return {}
    keys = set()
    for r in results_list:
        keys.update(r.keys())
    agg = {}
    for k in keys:
        vals = [r[k] for r in results_list if k in r and not np.isnan(r[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    return agg
