import numpy as np
from scipy.signal import savgol_filter


def _infer_axis(x, axis):
    if axis is not None:
        return int(axis)
    # Heuristic: prefer a 300-band axis, else the smallest axis.
    candidates = [i for i, s in enumerate(x.shape) if s == 300]
    return candidates[0] if candidates else int(np.argmin(x.shape))


def _normalize_window(bands, window_length, polyorder):
    # Make window_length odd, > polyorder, and <= bands
    min_needed = polyorder + 2
    if min_needed % 2 == 0:
        min_needed += 1
    wl = max(window_length, min_needed)
    wl = min(wl, bands if bands % 2 == 1 else bands - 1)
    if wl % 2 == 0:
        wl -= 1
    if wl <= polyorder:
        raise ValueError(
            f"window_length must be odd and > polyorder; got window_length={wl}, polyorder={polyorder}"
        )
    return wl


def savgol_smooth_cube_stream(
    cube,
    window_length: int = 17,
    polyorder: int = 3,
    deriv: int = 0,
    delta: float = 1.0,
    axis: int | None = None,
    mode: str = "interp",
    cval: float = 0.0,
    dtype_out=np.float32,
    tile_rows: int = 256,
    out: np.ndarray | None = None,
):
    """
    Savitzky–Golay smooth a (memmap) hyperspectral cube without loading it all.

    Parameters
    ----------
    cube : ndarray-like (supports slicing), shape (R,C,B) or (B,R,C) or (R,B,C)
    out  : optional output array/memmap, same shape as cube. If None and cube is writable,
           smoothing is written back in-place. If None and cube is read-only, raises.
    axis : spectral axis (0, 1, or 2). If None, inferred (prefers 300, else smallest dim).
    tile_rows : number of row slices processed per chunk (after reordering so spectral axis is last).

    Returns
    -------
    out : ndarray-like (the same object you passed in `out`, or `cube` if in-place)
    """
    x = np.asarray(cube)
    if x.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {x.shape}.")

    spec_axis = _infer_axis(x, axis)
    bands = x.shape[spec_axis]
    wl = _normalize_window(bands, window_length, polyorder)

    # Move axes so we have (rows, cols, bands)
    if spec_axis != 2:
        x_view = np.moveaxis(x, spec_axis, 2)
    else:
        x_view = x
    R, C, B = x_view.shape

    # Prepare output
    if out is None:
        if hasattr(cube, "flags") and hasattr(cube, "dtype") and getattr(cube, "flags").writeable:
            out = cube  # in-place write
        else:
            raise ValueError("Provide a writable `out` array/memmap when input is read-only.")
    y = out
    if spec_axis != 2:
        y_view = np.moveaxis(y, spec_axis, 2)
    else:
        y_view = y

    def _check_finite(tile):
        if not np.isfinite(tile).all():
            raise ValueError("Input contains NaN/Inf. Fill or mask invalid values before smoothing.")

    # Stream tiles along rows
    for r0 in range(0, R, tile_rows):
        r1 = min(r0 + tile_rows, R)
        tile = x_view[r0:r1, :, :]

        _check_finite(tile)

        tile_float = tile.astype(np.float32, copy=False)
        smoothed = savgol_filter(
            tile_float, window_length=wl, polyorder=polyorder,
            deriv=deriv, delta=delta, axis=2, mode=mode, cval=cval
        )

        if dtype_out is None:
            y_view[r0:r1, :, :] = smoothed
        else:
            y_view[r0:r1, :, :] = smoothed.astype(dtype_out, copy=False)

    return out
