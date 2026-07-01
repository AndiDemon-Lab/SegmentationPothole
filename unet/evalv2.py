#EVALUASI MODEL V2 - dengan Dice, IoU, Precision, Recall, F1, Params, FLOPs

import argparse
from pathlib import Path
import json
import sys
import torch
from torch.utils.data import DataLoader

# Handle both relative and absolute imports
try:
    from .models.unet_model import UNet
    from .data.dataset import PotholeDataset
    from .utils.utils import dice_coef, iou_coef
except ImportError:
    from models.unet_model import UNet
    from data.dataset import PotholeDataset
    from utils.utils import dice_coef, iou_coef

BASE = Path(__file__).resolve().parents[1]

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate UNet v2")
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

def compute_metrics_from_confusion(pred, target, threshold=0.5, eps=1e-6):
    """Calculate all metrics from confusion matrix (TP, FP, FN, TN)"""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()
    tn = ((1 - pred_bin) * (1 - target_bin)).sum()
    
    # Dice Score = 2*TP / (2*TP + FP + FN)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    
    # IoU = TP / (TP + FP + FN)
    iou = (tp + eps) / (tp + fp + fn + eps)
    
    # Precision = TP / (TP + FP)
    precision = (tp + eps) / (tp + fp + eps)
    
    # Recall = TP / (TP + FN)
    recall = (tp + eps) / (tp + fn + eps)
    
    # F1 Score = 2*TP / (2*TP + FP + FN) = Dice
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    
    return dice, iou, precision, recall, f1

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_flops(model, input_size=(1, 3, 512, 512), device='cpu'):
    """Calculate FLOPs using thop library"""
    try:
        from thop import profile, clever_format
        input_tensor = torch.randn(input_size).to(device)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    except ImportError:
        print("[warning] thop tidak terinstall. Install dengan: pip install thop")
        return "N/A", "N/A"
    except Exception as e:
        print(f"[warning] Gagal menghitung FLOPs: {e}")
        return "N/A", "N/A"

def evaluate(model, loader, device, amp_enabled=False):
    model.eval()
    tot_dice, tot_iou = 0.0, 0.0
    tot_prec, tot_rec, tot_f1 = 0.0, 0.0, 0.0
    tot_samples = 0
    
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(imgs)
            logits = _extract_logits(out)
            
            # Apply sigmoid for binary segmentation
            preds = torch.sigmoid(logits)
            
            # Compute all metrics from same confusion matrix
            batch_d, batch_i, batch_p, batch_r, batch_f1 = compute_metrics_from_confusion(preds, masks)
            
            b = imgs.size(0)
            tot_dice += float(batch_d.item()) * b
            tot_iou += float(batch_i.item()) * b
            tot_prec += float(batch_p.item()) * b
            tot_rec += float(batch_r.item()) * b
            tot_f1 += float(batch_f1.item()) * b
            tot_samples += b
    
    if tot_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    return (tot_dice / tot_samples, 
            tot_iou / tot_samples, 
            tot_prec / tot_samples,
            tot_rec / tot_samples,
            tot_f1 / tot_samples)

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

    # Hitung parameters
    total_params, trainable_params = count_parameters(model)
    print(f"[model] Total parameters: {total_params:,}")
    print(f"[model] Trainable parameters: {trainable_params:,}")

    # Hitung FLOPs
    print("[model] Menghitung FLOPs...")
    flops, params_str = calculate_flops(model, input_size=(1, 3, args.img_size, args.img_size), device=device)
    print(f"[model] FLOPs: {flops}")
    print(f"[model] Params (thop): {params_str}")

    list_dir = Path(args.list_dir)
    test_list = list_dir / "test.txt"
    if not test_list.exists():
        raise SystemExit("test.txt tidak ditemukan.")

    test_set = PotholeDataset(img_size=args.img_size, list_file=test_list)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers,
                             shuffle=False, pin_memory=(device.type == "cuda"))

    dice, iou, prec, rec, f1 = evaluate(model, test_loader, device, amp_enabled)
    
    print("\n" + "="*60)
    print("HASIL EVALUASI")
    print("="*60)
    print(f"Dice Score:    {dice:.4f}")
    print(f"IoU Score:     {iou:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"Total Params:  {total_params:,}")
    print(f"Train Params:  {trainable_params:,}")
    print(f"FLOPs:         {flops}")
    print("="*60)

    # Simpan hasil ke file
    results = {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "flops": str(flops),
        "model_info": {
            "bilinear": bilinear,
            "base_channels": base_ch,
            "img_size": args.img_size
        }
    }

    with open(logs_dir / "test_metrics_v2.txt", "w") as f:
        f.write(f"Dice Score:    {dice:.4f}\n")
        f.write(f"IoU Score:     {iou:.4f}\n")
        f.write(f"Precision:     {prec:.4f}\n")
        f.write(f"Recall:        {rec:.4f}\n")
        f.write(f"F1 Score:      {f1:.4f}\n")
        f.write(f"Total Params:  {total_params:,}\n")
        f.write(f"Train Params:  {trainable_params:,}\n")
        f.write(f"FLOPs:         {flops}\n")

    with open(logs_dir / "test_metrics_v2.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[eval] Hasil disimpan ke {logs_dir}")
    print("[eval] selesai.")

if __name__ == "__main__":
    main()
