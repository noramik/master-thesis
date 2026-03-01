import numpy as np

from rgb_utils import coerce_wavelengths


def ensure_rcb(cube, wavelengths):
    """Ensure shape is (rows, cols, bands) without copying."""
    B = len(wavelengths)
    if cube.shape[2] == B:  # (R,C,B)
        return cube
    if cube.shape[1] == B:  # (R,B,C)
        return np.moveaxis(cube, 1, 2)
    if cube.shape[0] == B:  # (B,R,C)
        return np.moveaxis(cube, 0, 2)
    raise ValueError(f"None of the axes match band count {B}. Got {cube.shape}.")


def slice_by_wavelength(cube, wavelengths, min_nm, max_nm):
    """
    Returns a view cropped to [min_nm, max_nm] on the band axis.
    No copy when the selected bands form a contiguous block.
    """
    wl = coerce_wavelengths(wavelengths)
    cube = ensure_rcb(cube, wl)

    # assume wl is sorted ascending (typical for sensors)
    i0 = int(np.searchsorted(wl, min_nm, side='left'))
    i1 = int(np.searchsorted(wl, max_nm, side='right'))
    if i0 >= i1:
        raise ValueError("Chosen wavelength window selects no bands.")

    cube_view = cube[:, :, i0:i1]    # view on the memmap (no copy)
    wl_view = wl[i0:i1]              # wavelength subset
    band_slice = slice(i0, i1)       # keep for bookkeeping
    return cube_view, wl_view, band_slice
