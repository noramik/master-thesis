import re
from collections import defaultdict

import numpy as np


def average_spectrum_stream(cube_hwb: np.ndarray,
                            mask_hw: np.ndarray,
                            mode: str = "raw",
                            eps: float = 1e-6,
                            tile_rows: int = 512):
    """
    Memory-efficient average spectrum for memmap cubes.
    cube_hwb : (H, W, B) hyperspectral cube (e.g., np.memmap)
    mask_hw  : (H, W) bool/0-1 mask
    mode     : "raw", "per_pixel_l2", or "per_pixel_zscore"
    eps      : avoid divide-by-zero
    tile_rows: process this many rows at a time
    Returns:
      spec_mean : (B,) float32
      spec_std  : (B,) float32
      n_pix     : int
    """
    H, W, B = cube_hwb.shape
    mask_hw = mask_hw.astype(bool)
    n_pix_total = 0

    sum_vec = np.zeros(B, np.float64)
    sumsq_vec = np.zeros(B, np.float64)

    for r0 in range(0, H, tile_rows):
        r1 = min(r0 + tile_rows, H)
        tile = cube_hwb[r0:r1, :, :].astype(np.float32, copy=False)
        mask_tile = mask_hw[r0:r1, :]

        # Extract only masked pixels
        Xm = tile[mask_tile]
        if Xm.size == 0:
            continue

        if mode == "raw":
            Xm_proc = Xm
        elif mode == "per_pixel_l2":
            norms = np.linalg.norm(Xm, axis=1, keepdims=True)
            Xm_proc = Xm / (norms + eps)
        elif mode == "per_pixel_zscore":
            mu = Xm.mean(axis=1, keepdims=True)
            sd = Xm.std(axis=1, keepdims=True)
            Xm_proc = (Xm - mu) / (sd + eps)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        sum_vec += Xm_proc.sum(axis=0, dtype=np.float64)
        sumsq_vec += np.square(Xm_proc, dtype=np.float64).sum(axis=0, dtype=np.float64)
        n_pix_total += Xm_proc.shape[0]

    if n_pix_total == 0:
        raise ValueError("Mask is empty; no pixels to average.")

    mean = (sum_vec / n_pix_total).astype(np.float32)
    var = (sumsq_vec / n_pix_total) - np.square(mean, dtype=np.float64)
    std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)

    return mean, std, n_pix_total


def average_by_treatment(sample_dict):
    """
    Groups keys by prefix before '_sampleX' and averages spectra.

    sample_dict : {str: np.ndarray}
        Keys like "treatA_sample1", values are (B,) mean spectra.
    Returns
    -------
    treat_dict : {str: np.ndarray}
        Keys like "treatA", values are (B,) averaged spectra.
    """
    groups = defaultdict(list)
    for key, spec in sample_dict.items():
        # strip suffix like "_sample1" or "_sample2"
        base = re.sub(r'_sample\d+.*$', '', key)
        groups[base].append(spec.astype(np.float64, copy=False))

    treat_dict = {}
    for base, specs in groups.items():
        stacked = np.vstack(specs)       # (n_samples, B)
        mean = stacked.mean(axis=0)      # (B,)
        treat_dict[base] = mean.astype(np.float32)

    return treat_dict
