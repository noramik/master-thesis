import numpy as np


def coerce_wavelengths(wavelengths):
    """
    Convert ENVI metadata 'wavelength' (strings or numbers) to a numpy array of nm.
    If values look like micrometers (< 20), convert to nm.
    """
    wl = []
    for w in wavelengths:
        try:
            wl.append(float(str(w).strip()))
        except Exception:
            pass
    wl = np.array(wl, dtype=float)

    # Heuristic: if max < 20, assume micrometers and convert to nm
    if wl.size and np.nanmax(wl) < 20:
        wl = wl * 1000.0
    return wl


def nearest_band_indices(wavelengths, targets_nm):
    """
    wavelengths: list/array of band center wavelengths (nm or µm; auto-detected)
    targets_nm:  list/tuple of target wavelengths in nm, e.g. [650, 550, 470]

    Returns:
      idx: list of nearest band indices (same order as targets)
      actual_nm: list of actual wavelength centers (nm) at those indices
    """
    wl = coerce_wavelengths(wavelengths)
    if wl.size == 0:
        raise ValueError("No usable wavelengths found.")

    targets_nm = np.array(targets_nm, dtype=float)
    idx = np.abs(wl[None, :] - targets_nm[:, None]).argmin(axis=1)
    actual = wl[idx].tolist()
    return idx.tolist(), actual


def rgb_from_bands(cube, band_indices, clip_percent=(2, 98)):
    """
    Build an RGB image from a hyperspectral cube and 3 band indices.
    - cube: (rows, cols, bands) np.ndarray or memmap-like
    - band_indices: [R_idx, G_idx, B_idx]
    - clip_percent: percentile clip per channel for contrast stretching
    Returns float32 RGB in [0,1].
    """
    r = cube[:, :, band_indices[0]].astype(np.float32)
    g = cube[:, :, band_indices[1]].astype(np.float32)
    b = cube[:, :, band_indices[2]].astype(np.float32)
    rgb = np.stack([r, g, b], axis=-1)

    # Per-channel percentile stretch -> [0,1]
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], clip_percent)
        if hi <= lo:
            out[..., c] = 0.0
        else:
            out[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo), 0, 1)
    return out
