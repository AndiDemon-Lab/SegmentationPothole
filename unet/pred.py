#PREDIKSI DATA SEKUNDER
from pathlib import Path
import random
import json
import numpy as np
from PIL import Image
import torch
import math

try:
    from .models.unet_model import UNet
    from .utils.utils import ensure_dir
except:
    try:
        from models.unet_model import UNet
        from utils.utils import ensure_dir
    except:
        def ensure_dir(p):
            Path(p).mkdir(parents=True, exist_ok=True)
        raise ImportError("Cannot import UNet model.")

HERE = Path(__file__).resolve().parent

IMG_DIR = (HERE / ".." / "data_testing3" / "image").resolve()
GT_MASK_DIR = (HERE / ".." / "data_testing3" / "mask").resolve()

OUT_BASE = (HERE / ".." / "result_train" / "output").resolve()
OUT_OVERLAYS = OUT_BASE / "overlays"
OUT_MASKS = OUT_BASE / "mask"
OUT_COLLAGE = OUT_BASE / "collage.png"
OUT_COLLAGE_MASKS_DIR = OUT_BASE / "collage_masks"
OUT_METRIC_JSON = OUT_BASE / "metrics.json"

CKPT_DIR = (HERE / ".." / "result_train" / "checkpoints").resolve()

IMG_SIZE = 512
THR = 0.5
TILE = 256
COLS = 5
NUM_EXAMPLES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_ckpt(d):
    a = d / "unet_last.pt"
    b = d / "unet_best.pt"
    if a.exists(): return a
    if b.exists(): return b
    pts = sorted(d.glob("*.pt"))
    return pts[-1] if pts else None

def gather_images(d):
    ex = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    p = Path(d)
    if not p.exists():
        return []
    return [f for f in sorted(p.rglob("*")) if f.suffix.lower() in ex]

def preprocess(p, size):
    im = Image.open(p).convert("RGB")
    orig = im.copy()
    proc = im.resize((size, size), Image.BILINEAR)
    arr = np.array(proc).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    t = torch.from_numpy(arr).unsqueeze(0)
    return orig, t

def mask_from_prob(prob, thr):
    m = prob.detach().cpu().numpy()
    b = (m > thr).astype("uint8") * 255
    return Image.fromarray(b, mode="L")

def create_overlay(orig, mask_l):
    img = orig.convert("RGBA")
    m = mask_l.resize(img.size, Image.NEAREST)
    ma = np.array(m)
    h, w = img.size[1], img.size[0]
    ov = np.zeros((h, w, 4), dtype=np.uint8)
    ov[..., 2] = 255
    ov[..., 3] = (ma / 255 * 90).astype(np.uint8)
    ov_img = Image.fromarray(ov, "RGBA")
    return Image.alpha_composite(img, ov_img).convert("RGB")

def make_triplet(orig, mask_l, overlay_img, tile):
    mask_rgb = Image.merge("RGB", (mask_l, mask_l, mask_l))
    orig_r = orig.resize((tile, tile))
    mask_r = mask_rgb.resize((tile, tile))
    ov_r = overlay_img.resize((tile, tile))
    comp = Image.new("RGB", (tile * 3, tile), (40, 40, 40))
    comp.paste(orig_r, (0, 0))
    comp.paste(mask_r, (tile, 0))
    comp.paste(ov_r, (tile * 2, 0))
    return comp

def make_collage(imgs, cols, tile):
    rows = math.ceil(len(imgs) / cols)
    W = cols * tile
    H = rows * tile
    canvas = Image.new("RGB", (W, H), (40, 40, 40))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas.paste(im.resize((tile, tile)), (c * tile, r * tile))
    return canvas

def compute_dice_iou(pred, gt, eps=1e-7):
    intersection = np.sum(pred * gt)
    dice = (2 * intersection + eps) / (np.sum(pred) + np.sum(gt) + eps)
    iou = (intersection + eps) / (np.sum((pred + gt) > 0) + eps)
    return float(dice), float(iou)

def main():
    ensure_dir(OUT_BASE)
    ensure_dir(OUT_OVERLAYS)
    ensure_dir(OUT_MASKS)
    ensure_dir(OUT_COLLAGE_MASKS_DIR)

    ckpt = find_ckpt(CKPT_DIR)
    if ckpt is None:
        return

    model = UNet(n_channels=3, n_classes=1, bilinear=True, base_c=64).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=False)
    model.eval()

    files = gather_images(IMG_DIR)
    examples = random.sample(files, min(NUM_EXAMPLES, len(files)))

    saved_example_overlays = []
    metrics = {}
    dice_all, iou_all = [], []

    with torch.no_grad():
        for f in files:
            orig, tensor = preprocess(f, IMG_SIZE)
            logits = model(tensor.to(DEVICE))
            probs = torch.sigmoid(logits)[0, 0]

            mask_l = mask_from_prob(probs, THR)
            overlay_img = create_overlay(orig, mask_l)

            pred_mask_path = OUT_MASKS / f"{f.stem}_predmask.png"
            overlay_path = OUT_OVERLAYS / f"{f.stem}_overlay.png"

            mask_l.resize(orig.size, Image.NEAREST).save(pred_mask_path)
            overlay_img.save(overlay_path)

            gt_path = GT_MASK_DIR / f"{f.stem}_mask.png"
            if gt_path.exists():
                pred = np.array(mask_l.resize(orig.size)) > 0
                gt = np.array(Image.open(gt_path).convert("L")) > 0

                dice, iou = compute_dice_iou(pred, gt)
                metrics[f.stem] = {
                    "dice": round(dice, 4),
                    "iou": round(iou, 4)
                }
                dice_all.append(dice)
                iou_all.append(iou)

            if f in examples:
                saved_example_overlays.append(overlay_img)
                trip = make_triplet(orig, mask_l, overlay_img, TILE)
                trip.save(OUT_COLLAGE_MASKS_DIR / f"{f.stem}_collage.png")

    if saved_example_overlays:
        col = make_collage(saved_example_overlays, COLS, TILE)
        col.save(OUT_COLLAGE)

    if dice_all:
        metrics["summary"] = {
            "mean_dice": round(float(np.mean(dice_all)), 4),
            "mean_iou": round(float(np.mean(iou_all)), 4),
            "num_samples": len(dice_all)
        }

    with open(OUT_METRIC_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
