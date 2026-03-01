import numpy as np


def apply_white_reference_memmap(
    cube: np.ndarray,
    y_range: str | tuple[int, int],
    x_range: str | tuple[int, int],
    *,
    eps: float = 1e-12,
    tile_rows: int = 256,
    dtype_out=np.float32,
    out: np.ndarray | None = None,
):
    """
    White-reference a hyperspectral cube (R, C, B) using a white patch.

    Parameters
    ----------
    cube : ndarray-like (can be a memmap), shape (rows, cols, bands).
    y_range : 'y0:y1' or (y0, y1)  -- rows covering the white patch.
    x_range : 'x0:x1' or (x0, x1)  -- cols covering the white patch.
    eps : float
        Avoid divide-by-zero if patch mean has zeros.
    tile_rows : int
        Process this many rows at a time (keeps memory low).
    dtype_out : dtype, optional
        Output dtype. Use float32 for reflectance.
    out : ndarray-like, optional
        Writable target array (same shape as cube). If None and `cube` is
        writable, modifies in-place.

    Returns
    -------
    out : ndarray-like
        The white-referenced cube (same object as `out` if provided).
    """
    R, C, B = cube.shape

    # Parse ranges
    if isinstance(y_range, str):
        y0, y1 = map(int, y_range.split(':'))
    else:
        y0, y1 = y_range
    if isinstance(x_range, str):
        x0, x1 = map(int, x_range.split(':'))
    else:
        x0, x1 = x_range

    if not (0 <= y0 < y1 <= R) or not (0 <= x0 < x1 <= C):
        raise ValueError("Invalid patch coordinates")

    # Compute mean spectrum from the patch
    patch = cube[y0:y1, x0:x1, :]                # view on memmap
    mean_spec = patch.astype(np.float32, copy=False).mean(axis=(0, 1))
    mean_spec = np.where(np.isfinite(mean_spec), mean_spec, 0.0)
    mean_spec = np.where(mean_spec == 0.0, eps, mean_spec).astype(np.float32)

    # Prepare output
    if out is None:
        if hasattr(cube, "flags") and cube.flags.writeable:
            out = cube
        else:
            raise ValueError("Cube is read-only; provide a writable `out`.")
    if out.shape != cube.shape:
        raise ValueError("`out` must have same shape as cube")

    # Stream rows
    for r0 in range(0, R, tile_rows):
        r1 = min(r0 + tile_rows, R)
        tile = cube[r0:r1, :, :].astype(np.float32, copy=False)
        corrected = tile / mean_spec[np.newaxis, np.newaxis, :]
        if dtype_out is not None and np.dtype(dtype_out) != corrected.dtype:
            corrected = corrected.astype(dtype_out, copy=False)
        out[r0:r1, :, :] = corrected

    return out
