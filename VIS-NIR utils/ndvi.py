import numpy as np


def band_index_for_wavelength(wavelengths, target_nm):
    """
    Find the band index closest to target_nm.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    return int(np.argmin(np.abs(wavelengths - target_nm)))


def ndvi_mask(cube_hwb, wavelengths, red_nm=670.0, nir_nm=800.0, thresh=0.25):
    """
    Compute NDVI and threshold to get a vegetation mask.

    Args:
        cube_hwb: np.ndarray (H, W, B)
        wavelengths: list/np.ndarray of band centers [B]
        red_nm: wavelength to use for Red (≈670 nm)
        nir_nm: wavelength to use for NIR (≈800 nm)
        thresh: NDVI threshold for vegetation

    Returns:
        ndvi: float32 array (H, W) with NDVI values
        mask: bool array (H, W) where True = vegetation
    """
    red_idx = band_index_for_wavelength(wavelengths, red_nm)
    nir_idx = band_index_for_wavelength(wavelengths, nir_nm)

    red = cube_hwb[..., red_idx].astype(np.float32)
    nir = cube_hwb[..., nir_idx].astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-6)
    mask = ndvi > thresh
    return ndvi, mask
