import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def _build_idx(ann_path: Path):
    data = json.loads(ann_path.read_text())
    images_by_id = {im["id"]: im for im in data["images"]}
    anns_by_image: Dict[int, List[dict]] = {im_id: [] for im_id in images_by_id}
    for a in data["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    return images_by_id, anns_by_image, cat_id_to_name


def _polylist_to_mask(polys: List[List[float]], hw: Tuple[int, int]) -> np.ndarray:
    """COCO polygon(s) -> binary mask (HxW, uint8 in {0,1})."""
    h, w = hw
    m = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(m)
    for p in polys:
        if len(p) >= 6:
            pts = [(p[i], p[i + 1]) for i in range(0, len(p), 2)]
            draw.polygon(pts, outline=1, fill=1)
    return np.array(m, dtype=np.uint8)


def _scale_polys(polys: List[List[float]], sx: float, sy: float) -> List[List[float]]:
    """Scale polygon coordinate lists in-place style (return new list)."""
    out = []
    for p in polys:
        q = []
        for i, v in enumerate(p):
            q.append(v * (sx if i % 2 == 0 else sy))
        out.append(q)
    return out


class COCOSemantic4ClassHSI(Dataset):
    """
    Use COCO polygons for masks; load HSI from .npy by matching stem names.

    Classes: 0=background, 1=leaf, 2=label/tag, 3=soil

    Arguments:
      coco_split_dir: path containing *_annotations.coco.json (e.g., root/train/)
      hsi_dir: folder where <stem>.npy cubes live
      img_size: optional (H,W) to resize both cube and mask (None keeps native HSI size)
      leaf_names/label_names/soil_names: category names mapped to classes 1/2/3
      mmap: use np.load(..., mmap_mode='r') for lower RAM if large
    """

    def __init__(
        self,
        coco_split_dir: Path,
        hsi_dir: Path,
        img_size: Optional[Tuple[int, int]] = (480, 480),
        leaf_names=('strawberry-leaf', 'leaf', 'leaves', 'strawberry-leaves'),
        label_names=('label', 'tag'),
        soil_names=('soil', 'dirt', 'ground'),
        mmap: bool = False,
    ):
        self.split_dir = Path(coco_split_dir)
        self.hsi_dir   = Path(hsi_dir)
        self.img_size  = img_size

        # COCO json
        self.ann_path = self.split_dir / "_annotations.coco.json"
        if not self.ann_path.exists():
            alt = self.split_dir / "annotations.coco.json"
            if alt.exists():
                self.ann_path = alt
            else:
                raise FileNotFoundError(f"COCO annotations not found in {self.split_dir}")

        self.images_by_id, self.anns_by_image, self.cat_id_to_name = _build_idx(self.ann_path)

        # Map category_id -> class_index {background:0, leaf:1, label:2, soil:3}
        leaf_set = {n.lower() for n in leaf_names}
        labl_set = {n.lower() for n in label_names}
        soil_set = {n.lower() for n in soil_names}

        self.cat_to_class: Dict[int, int] = {}
        for cid, name in self.cat_id_to_name.items():
            n = name.lower()
            if n in leaf_set:
                self.cat_to_class[cid] = 1
            elif n in labl_set:
                self.cat_to_class[cid] = 2
            elif n in soil_set:
                self.cat_to_class[cid] = 3
            # else -> background (0)

        if not any(v in (1, 2, 3) for v in self.cat_to_class.values()):
            raise ValueError(
                "No matching categories found for leaf/label/soil. "
                f"Available categories: {sorted(self.cat_id_to_name.values())}"
            )

        # Build stem -> HSI npy path map
        hsi_paths = {p.stem: p for p in self.hsi_dir.glob("*.npy")}

        # Build dataset items: (hsi_path, anns_for_image, rgb_size)
        self.items: List[Tuple[Path, List[dict], Tuple[Optional[int], Optional[int]]]] = []
        missing_hsi = []
        for im_id, im in self.images_by_id.items():
            stem = Path(im["file_name"]).stem
            hsi_path = hsi_paths.get(stem)
            if hsi_path is None:
                missing_hsi.append(stem)
                continue
            rgb_w = im.get("width")
            rgb_h = im.get("height")
            if rgb_w is None or rgb_h is None:
                rgb_w = rgb_h = None
            self.items.append((hsi_path, self.anns_by_image.get(im_id, []), (rgb_h, rgb_w)))

        if not self.items:
            hint = "\n  - " + "\n  - ".join(missing_hsi[:10])
            raise RuntimeError(
                f"No HSI .npy matched COCO file_name stems in {self.hsi_dir}.\n"
                f"Examples of missing stems:{hint}"
            )

        self.mmap = mmap

        # Overwrite priority: higher wins when polygons overlap
        # background (0) < soil (1) < leaf/tag (3)
        self.priority = {0: 0, 1: 3, 2: 3, 3: 1}

    def __len__(self):
        return len(self.items)

    def _load_hsi(self, path: Path) -> np.ndarray:
        arr = np.load(path, mmap_mode='r' if self.mmap else None)
        # Ensure float32 and (C,H,W)
        if arr.ndim == 3:
            # Accept (H,W,C) or (C,H,W)
            if arr.shape[-1] < min(arr.shape[0], arr.shape[1]):
                # (H,W,C) where C is smaller than H,W
                arr = np.transpose(arr, (2, 0, 1))
            elif arr.shape[0] < min(arr.shape[1], arr.shape[2]):
                # already (C,H,W)
                pass
            else:
                # ambiguous; default to (H,W,C) -> (C,H,W)
                arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"Expected 3D HSI cube, got shape {arr.shape} for {path}")
        return arr.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        hsi_path, anns, (rgb_h, rgb_w) = self.items[idx]

        cube = self._load_hsi(hsi_path)            # (C,H,W), float32
        C, H, W = cube.shape

        # Scale polygons to HSI size if RGB and HSI differ
        if rgb_w is not None and rgb_h is not None and (rgb_w != W or rgb_h != H):
            sx = W / float(rgb_w)
            sy = H / float(rgb_h)
        else:
            sx = sy = 1.0

        # Create mask at HSI native size
        mask = np.zeros((H, W), dtype=np.uint8)
        for a in anns:
            cid = a.get("category_id")
            cls = self.cat_to_class.get(cid, 0)
            if cls == 0:
                continue
            seg = a.get("segmentation")
            if seg is None:
                continue
            if isinstance(seg, list):
                seg_scaled = _scale_polys(seg, sx, sy) if (sx != 1.0 or sy != 1.0) else seg
                m = _polylist_to_mask(seg_scaled, (H, W))  # 0/1
                if m.any():
                    # Overwrite with priority: new class must have >= current priority
                    cur = mask
                    cur_pri = np.vectorize(self.priority.get)(cur)
                    new_pri = self.priority.get(cls, 0)
                    write = (m == 1) & (new_pri >= cur_pri)
                    mask[write] = cls
            elif isinstance(seg, dict) and "counts" in seg:
                raise RuntimeError("RLE segmentation encountered. Decode RLE or re-export polygons.")

        # To torch tensors
        x = torch.from_numpy(cube)                 # (C,H,W), float32
        y = torch.from_numpy(mask.astype(np.int64))# (H,W), long for CE loss

        # Optional resize to training size
        if self.img_size is not None:
            th, tw = self.img_size
            x = F.interpolate(x.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False).squeeze(0)
            y = F.interpolate(y.unsqueeze(0).unsqueeze(0).float(), size=(th, tw), mode="nearest").squeeze(0).squeeze(0).long()

        return x, y, hsi_path.stem  # also returning stem helps debugging/saving


def make_loaders_coco_hsi(
    coco_root: str,          # folder containing train/ and valid/ with COCO jsons
    hsi_root: str,           # folder with <stem>.npy cubes
    batch_size=2,
    img_size=(480, 480),
    leaf_names=('strawberry-leaf', 'leaf', 'leaves', 'strawberry-leaves'),
    label_names=('label', 'tag'),
    soil_names=('soil', 'dirt', 'ground'),
    num_workers=0,
    mmap=False,
):
    coco_root = Path(coco_root)
    tr_dir = coco_root / "train"
    va_dir = coco_root / "valid"
    if not tr_dir.exists() or not va_dir.exists():
        raise FileNotFoundError(f"Expected 'train' and 'valid' under {coco_root}")

    train_ds = COCOSemantic4ClassHSI(
        tr_dir, hsi_root, img_size=img_size,
        leaf_names=leaf_names, label_names=label_names, soil_names=soil_names, mmap=mmap
    )
    val_ds = COCOSemantic4ClassHSI(
        va_dir, hsi_root, img_size=img_size,
        leaf_names=leaf_names, label_names=label_names, soil_names=soil_names, mmap=mmap
    )

    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
    va = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return tr, va
