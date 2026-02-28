import numpy as np
import cv2


def postprocess_binary_mask(mask_bool: np.ndarray,
                            open_ks: int = 3,
                            close_ks: int = 5,
                            min_area: int = 800,
                            fill_holes: bool = True,
                            max_hole_area: int | None = 5000) -> np.ndarray:
    """
    mask_bool : (H,W) bool/0-1 vegetation mask (e.g., NDVI > τ)
    open_ks   : kernel for opening (remove speckles). 0/1 -> skip
    close_ks  : kernel for closing (smooth edges, bridge small gaps). 0/1 -> skip
    min_area  : drop components smaller than this (px)
    fill_holes: fill only interior holes (do not touch border)
    max_hole_area : if not None, only fill holes up to this area
    Returns: (H,W) uint8 {0,1}
    """
    m = (mask_bool.astype(np.uint8) > 0).astype(np.uint8)

    # 1) Opening then Closing (less likely to merge big regions)
    if open_ks and open_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_ks and close_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    # 2) Remove small components
    if min_area and min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(num, dtype=bool)
        for i in range(1, num):
            keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
        m = keep[labels].astype(np.uint8)

    # 3) Fill interior holes (components of the inverse that don't touch border)
    if fill_holes:
        inv = (1 - m).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        H, W = m.shape
        for i in range(1, num):
            x, y, w, h, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
            touches_border = (x == 0) or (y == 0) or (x + w == W) or (y + h == H)
            if not touches_border:
                if (max_hole_area is None) or (area <= max_hole_area):
                    m[labels == i] = 1

    return m
