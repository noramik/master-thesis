import torch, torch.nn as nn
from coco_loaders import make_loaders_coco_hsi
from unet_hsi import UNetPlusPlus  # must accept in_bands + num_classes

def tensor_is_images(t):
    # Likely images: [B, C, H, W] float/half
    return torch.is_tensor(t) and t.ndim == 4 and t.shape[1] >= 1 and t.dtype.is_floating_point

def tensor_is_masks(t):
    # Likely masks: [B, H, W] int OR [B, C, H, W] one-hot/binary
    if not torch.is_tensor(t) or t.ndim < 3:
        return False
    if t.ndim == 3 and t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return True
    if t.ndim == 4 and (t.dtype == torch.float32 or t.dtype == torch.float16 or t.dtype == torch.uint8):
        return True
    return False

def to_class_ids(mask):
    """
    Accepts:
      - [B, H, W] ints in {0..C-1}  -> returns same (long)
      - [B, C, H, W] one-hot/soft   -> argmax over channel -> [B, H, W] long
      - [B, 1, H, W] binary         -> squeeze -> [B, H, W] long
    """
    if mask.ndim == 3:
        return mask.long()
    if mask.ndim == 4:
        if mask.shape[1] == 1:
            return mask[:, 0].long()
        # assume channel-first one-hot/prob
        return mask.argmax(dim=1).long()
    raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")

def pick_img_mask(batch):
    """
    Robustly pull (images, masks) from many batch formats:
    - dict with keys like image/images & mask/masks/target/targets/labels
    - tuple/list with (img, mask, *extras)
    - tuple/list arbitrary order: we detect by tensor shapes/dtypes
    """
    # dict case
    if isinstance(batch, dict):
        candidates_img = [k for k in batch.keys() if k.lower() in ("image","images","x","inputs")]
        candidates_msk = [k for k in batch.keys() if k.lower() in ("mask","masks","y","target","targets","labels")]
        img = None; msk = None
        for k in candidates_img:
            if tensor_is_images(batch[k]):
                img = batch[k]; break
        for k in candidates_msk:
            if tensor_is_masks(batch[k]):
                msk = batch[k]; break
        if img is None or msk is None:
            # fallback: scan all values
            for v in batch.values():
                if img is None and tensor_is_images(v): img = v
                elif msk is None and tensor_is_masks(v): msk = v
        if img is None or msk is None:
            raise ValueError(f"Could not find image/mask in dict batch keys: {list(batch.keys())}")
        return img, to_class_ids(msk)

    # tuple/list case
    if isinstance(batch, (tuple, list)):
        img = None; msk = None
        for item in batch:
            if img is None and tensor_is_images(item):
                img = item
            elif msk is None and tensor_is_masks(item):
                msk = item
        if img is None or msk is None:
            raise ValueError(f"Could not find image/mask in tuple/list batch of length {len(batch)}")
        return img, to_class_ids(msk)

    # unknown
    raise TypeError(f"Unsupported batch type: {type(batch)}")

def main():
    coco_root = r"D:\hyperspectral\image_segm_deep_learning\strawberry_annotated3"
    hsi_root  = r"D:\hyperspectral\dataset_npy"

    # --- Data ---
    tr, va = make_loaders_coco_hsi(
        coco_root, hsi_root,
        batch_size=4, img_size=(480, 480),
        leaf_names=('strawberry-leaf', 'leaf', 'leaves'),
        label_names=('label',),
        soil_names=('soil','dirt','ground')
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Detect HSI bands
    try:
        in_bands = getattr(tr.dataset, "num_bands", None)
        if in_bands is None:
            first_batch = next(iter(tr))
            imgs0, _ = pick_img_mask(first_batch)
            in_bands = imgs0.shape[1]
        print(f"Detected hyperspectral bands: {in_bands}")
    except Exception:
        in_bands = 235  # fallback
        print(f"Using configured hyperspectral bands: {in_bands}")

    NUM_CLASSES = 4
    model = UNetPlusPlus(in_bands=in_bands, num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    epochs = 50
    best_val = float("inf")
    #best_path = "unetpp_strawberry_4class_best.pth"

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for i, batch in enumerate(tr):
            imgs, masks = pick_img_mask(batch)
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(imgs)               # [B,3,H,W]
                loss = criterion(logits, masks)    # masks: [B,H,W] long
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            if i % 5 == 0:
                print(f"  batch {i}/{len(tr)}  loss {loss.item():.4f}")
        tr_loss = running / max(1, len(tr))

        # -- Validation --
        model.eval()
        val_sum = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            for batch in va:
                imgs, masks = pick_img_mask(batch)
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(imgs)
                val_sum += criterion(logits, masks).item()
        va_loss = val_sum / max(1, len(va))

        print(f"Epoch {ep}/{epochs}  train:{tr_loss:.4f}  val:{va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best val {best_val:.4f}. Saved to {best_path}")

    final_path = "unetpp_strawberry_4class_last.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final weights to {final_path} (best was {best_path} with {best_val:.4f})")

if __name__ == "__main__":
    main()

   
