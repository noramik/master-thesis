# predict_hsi.py
import os, glob
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

# ----- Your model -----
from unet_hsi import UNetPlusPlus  # must accept (in_bands, num_classes)


# ===================== CONFIG =====================
IMG_SIZE: Tuple[int, int] = (480, 480)   # (H, W) exactly as in training
INPUT_DIR = r"D:\hyperspectral\SWIR\dataset_npy"       # folder with .npy files
OUTPUT_DIR = r"D:\hyperspectral\Nora\predictions_hsi_new"
CKPT_PATH = r"D:\hyperspectral\unetpp_strawberry_4class_best.pth"

NUM_CLASSES = 4
CLASS_NAMES: List[str] = ["background", "strawberry-leaf", "label", "soil"]

# Colors for visualization (RGB 0-255)
PALETTE = {
    0: (0, 0, 0),        # background
    1: (0, 200, 0),      # leaf
    2: (220, 0, 0),      # label
    3: (255,164,0)       # soil
}

# Which class ids count as "plant" for the binary mask
PLANT_CLASS_IDS = (1,)   # change to (1,3) if you ever want leaf+soil, etc.

# Normalization
MEAN_STD_PATH: Optional[str] = None  # e.g., r"D:\hyperspectral\train_stats.npz" or .npy
NORM_MODE: str = "none"  # "zscore" (per-image) or "none"

# RGB preview options
RGB_BAND_INDICES: Optional[Tuple[int, int, int]] = None  # e.g., (30, 20, 10)
RGB_PERCENTILE_STRETCH: Tuple[float, float] = (1.0, 99.0)
# ==================================================


def load_mean_std(path):
    if path is None or not os.path.exists(path):
        return None, None
    if path.endswith(".npz"):
        z = np.load(path)
        mean = z["mean"]; std = z["std"]
    else:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr.item(), dict):
            d = arr.item()
            mean, std = d["mean"], d["std"]
        else:
            mean, std = arr[0], arr[1]
    return mean.astype(np.float32), std.astype(np.float32)


def ensure_hwb(cube: np.ndarray) -> np.ndarray:
    """Ensure cube shape is (H, W, B). Accepts (H,W,B) or (B,H,W)."""
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D HSI cube, got shape {cube.shape}")
    if cube.shape[0] < 16 and cube.shape[1] > 32 and cube.shape[2] > 32:
        cube = np.transpose(cube, (1, 2, 0))
    elif cube.shape[2] < 16 and cube.shape[0] > 32 and cube.shape[1] > 32:
        pass
    elif cube.shape[2] >= 16 and cube.shape[0] > 32 and cube.shape[1] > 32:
        pass
    else:
        cube = np.transpose(cube, (1, 2, 0))
    return cube


def normalize_hsi(cube_hwb: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    X = cube_hwb.astype(np.float32)
    if mean is not None and std is not None and mean.ndim == 1 and mean.shape[0] == X.shape[2]:
        return (X - mean[None, None, :]) / (std[None, None, :] + 1e-6)
    if NORM_MODE.lower() == "zscore":
        mu = np.mean(X, axis=(0, 1), keepdims=True)
        si = np.std(X, axis=(0, 1), keepdims=True)
        return (X - mu) / (si + 1e-6)
    return X


def get_rgb(cube_hwb: np.ndarray,
            band_indices: Optional[Tuple[int, int, int]] = RGB_BAND_INDICES,
            pct: Tuple[float, float] = RGB_PERCENTILE_STRETCH) -> Image.Image:
    H, W, B = cube_hwb.shape
    if band_indices is None:
        r = int(B * 0.75) - 1
        g = int(B * 0.50) - 1
        b = int(B * 0.25) - 1
        band_indices = (max(0, min(B-1, r)),
                        max(0, min(B-1, g)),
                        max(0, min(B-1, b)))
    r_idx, g_idx, b_idx = band_indices
    rgb = np.stack([cube_hwb[:, :, r_idx],
                    cube_hwb[:, :, g_idx],
                    cube_hwb[:, :, b_idx]], axis=-1).astype(np.float32)

    lo, hi = np.percentile(rgb, pct[0]), np.percentile(rgb, pct[1])
    if hi <= lo:
        lo, hi = rgb.min(), rgb.max()
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(rgb_u8, mode="RGB")


def colorize(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, color in PALETTE.items():
        rgb[mask == k] = color
    return Image.fromarray(rgb)


def overlay(base_img_pil: Image.Image, mask_rgb_pil: Image.Image, alpha=0.5) -> Image.Image:
    base = base_img_pil.convert("RGBA")
    mask = mask_rgb_pil.convert("RGBA").resize(base.size, Image.NEAREST)
    return Image.blend(base, mask, alpha)


def load_model(device, in_bands: int):
    model = UNetPlusPlus(in_bands=in_bands, num_classes=NUM_CLASSES).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    sd = state if isinstance(state, dict) else state.state_dict()
    # Remap any 'ds_*.0.*' -> 'ds_*.*'
    new_sd = {}
    for k, v in sd.items():
        if isinstance(k, str) and k.startswith("ds_") and ".0." in k:
            new_sd[k.replace(".0.", ".")] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def segment_hsi_multiclass(model, cube_hwb: np.ndarray, device,
                           mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    H, W, B = cube_hwb.shape
    X = normalize_hsi(cube_hwb, mean, std).astype(np.float32)
    xt = torch.from_numpy(np.transpose(X, (2, 0, 1))).unsqueeze(0).to(device)  # [1,B,H,W]
    xt_resized = F.interpolate(xt, size=IMG_SIZE, mode="bilinear", align_corners=False)
    logits = model(xt_resized)  # [1,C,h,w]
    pred_small = torch.argmax(logits, dim=1).squeeze(0)  # [h,w]
    pred = F.interpolate(pred_small.unsqueeze(0).unsqueeze(0).float(),
                         size=(H, W), mode="nearest").squeeze().to(torch.uint8).cpu().numpy()
    return pred


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npy")))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {INPUT_DIR}")

    first = ensure_hwb(np.load(paths[0], allow_pickle=True))
    H0, W0, B = first.shape
    print(f"Detected first cube: {H0}x{W0} with {B} bands")

    mean, std = load_mean_std(MEAN_STD_PATH)
    if mean is not None and std is not None:
        print("Loaded dataset mean/std for normalization.")
    else:
        print(f"Normalization mode: {NORM_MODE}")

    model = load_model(device, in_bands=B)

    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        cube = ensure_hwb(np.load(p, allow_pickle=True))
        H, W, B_ = cube.shape
        if B_ != B:
            raise ValueError(f"Band count mismatch: first file has {B}, but {p} has {B_}")

        # Predict class-index mask
        pred_mask = segment_hsi_multiclass(model, cube, device, mean, std)

        # --- NEW: build and save binary plant mask (.npy) ---
        # plant = any of PLANT_CLASS_IDS (default: leaf only = class 1)
        plant_mask = np.isin(pred_mask, PLANT_CLASS_IDS).astype(np.uint8)  # 1=plant, 0=non-plant
        plant_mask_path = os.path.join(OUTPUT_DIR, f"{base}_plantmask.npy")
        np.save(plant_mask_path, plant_mask)

        # Save color viz + overlay 
        rgb_img = get_rgb(cube)
        mask_color = colorize(pred_mask)
        mask_color_path = os.path.join(OUTPUT_DIR, f"{base}_mask_color.png")
        mask_color.save(mask_color_path)

        ov = overlay(rgb_img, mask_color, alpha=0.35)
        ov_path = os.path.join(OUTPUT_DIR, f"{base}_overlay.png")
        ov.save(ov_path)

        print(f"Saved: {plant_mask_path} | {mask_color_path} | {ov_path}")

    # legend
    legend_txt = os.path.join(OUTPUT_DIR, "_classes.txt")
    with open(legend_txt, "w", encoding="utf-8") as f:
        for i, name in enumerate(CLASS_NAMES[:NUM_CLASSES]):
            f.write(f"{i}: {name}\n")
    print(f"Wrote class legend to {legend_txt}")


if __name__ == "__main__":
    main()

