import argparse
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader

from .models.unet_model import UNet
from .data.dataset import PotholeDataset

BASE = Path(__file__).resolve().parents[1]

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate UNet (Precision, Recall, F1)")
    ap.add_argument("--mode", type=str, choices=["last", "best"], default="last")
    ap.add_argument("--list-dir", type=str, default=str(BASE / "dataset_split"))
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default=str(BASE / "result_train"))
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--bilinear", action="store_true", default=True)
    ap.add_argument("--threshold", type=float, default=0.5)
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

def evaluate_prf(model, loader, device, threshold=0.5, amp_enabled=False):
    model.eval()
    tp = fp = fn = 0.0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device).float()

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = _extract_logits(model(imgs))
                probs = torch.sigmoid(logits)

            preds = (probs > threshold).float()

            tp += torch.sum(preds * masks).item()
            fp += torch.sum(preds * (1 - masks)).item()
            fn += torch.sum((1 - preds) * masks).item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    ensure_dir(logs_dir)

    if args.mode == "last":
        ckpt_path = ckpt_dir / "unet_last.pt"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / "unet_best.pt"
    else:
        ckpt_path = ckpt_dir / "unet_best.pt"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / "unet_last.pt"

    if not ckpt_path.exists():
        raise SystemExit("[eval] checkpoint tidak ditemukan.")

    ckpt = load_checkpoint_full(ckpt_path, device)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    bilinear = ckpt.get("bilinear", args.bilinear) if isinstance(ckpt, dict) else args.bilinear
    base_ch = ckpt.get("base_ch", args.base_ch) if isinstance(ckpt, dict) else args.base_ch

    model = UNet(n_channels=3, n_classes=1, bilinear=bilinear, base_c=base_ch).to(device)
    model.load_state_dict(state_dict, strict=False)

    test_list = Path(args.list_dir) / "test.txt"
    if not test_list.exists():
        raise SystemExit("test.txt tidak ditemukan.")

    test_set = PotholeDataset(img_size=args.img_size, list_file=test_list)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    precision, recall, f1 = evaluate_prf(
        model,
        test_loader,
        device,
        threshold=args.threshold,
        amp_enabled=amp_enabled
    )

    print(f"[eval-final] precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")

    with open(logs_dir / "test_prf.json", "w") as f:
        json.dump(
            {"precision": precision, "recall": recall, "f1": f1},
            f,
            indent=2
        )

    print("[eval] selesai.")

if __name__ == "__main__":
    main()
