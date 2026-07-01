#EVALUASI MODEL

import argparse
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader

from .models.unet_model import UNet
from .data.dataset import PotholeDataset
from .utils.utils import dice_coef, iou_coef

BASE = Path(__file__).resolve().parents[1]

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate UNet")
    ap.add_argument("--mode", type=str, choices=["last", "best"], default="last")
    ap.add_argument("--list-dir", type=str, default=str(BASE / "dataset_split"))
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default=str(BASE / "result_train"))
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--bilinear", action="store_true", default=True)
    ap.add_argument("--amp", action="store_true")
    return ap.parse_args()

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_checkpoint_full(path: Path, device):
    return torch.load(path, map_location=device)

def _extract_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output

def evaluate(model, loader, device, amp_enabled=False):
    model.eval()
    tot_dice, tot_iou = 0.0, 0.0
    tot_samples = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(imgs)
            logits = _extract_logits(out)
            b = imgs.size(0)
            batch_d = float(dice_coef(logits, masks).item())
            batch_i = float(iou_coef(logits, masks).item())
            tot_dice += batch_d * b
            tot_iou += batch_i * b
            tot_samples += b
    if tot_samples == 0:
        return 0.0, 0.0
    return tot_dice / tot_samples, tot_iou / tot_samples

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    ensure_dir(ckpt_dir)
    ensure_dir(logs_dir)

    if args.mode == "last":
        c = ckpt_dir / "unet_last.pt"
        c2 = ckpt_dir / "unet_best.pt"
        ckpt_path = c if c.exists() else (c2 if c2.exists() else None)
    else:
        c = ckpt_dir / "unet_best.pt"
        c2 = ckpt_dir / "unet_last.pt"
        ckpt_path = c if c.exists() else (c2 if c2.exists() else None)

    if ckpt_path is None:
        raise SystemExit("[eval] Tidak ada checkpoint tersedia.")
    print(f"[eval] menggunakan checkpoint: {ckpt_path.name}")

    ckpt = load_checkpoint_full(ckpt_path, device)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    bilinear = ckpt.get("bilinear", args.bilinear) if isinstance(ckpt, dict) else args.bilinear
    base_ch = ckpt.get("base_ch", args.base_ch) if isinstance(ckpt, dict) else args.base_ch

    model = UNet(n_channels=3, n_classes=1, bilinear=bilinear, base_c=base_ch).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    list_dir = Path(args.list_dir)
    test_list = list_dir / "test.txt"
    if not test_list.exists():
        raise SystemExit("test.txt tidak ditemukan.")

    test_set = PotholeDataset(img_size=args.img_size, list_file=test_list)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers,
                             shuffle=False, pin_memory=(device.type == "cuda"))

    dice, iou = evaluate(model, test_loader, device, amp_enabled)
    print(f"[eval-final] dice={dice:.4f}  iou={iou:.4f}")

    with open(logs_dir / "test_metrics.txt", "w") as f:
        f.write(f"dice={dice:.4f}\niou={iou:.4f}\n")
    with open(logs_dir / "test_metrics.json", "w") as f:
        json.dump({"dice": dice, "iou": iou}, f, indent=2)

    print("[eval] selesai.")

if __name__ == "__main__":
    main()
