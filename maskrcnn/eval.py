from pathlib import Path
import json

import torch
import torchvision.transforms as T

from dataloader.getdata import get_dataloaders
from model.maskrcnn import get_instance_segmentation_model, load_model_checkpoint
from utils.utils import (
    ensure_dir,
    binary_iou,
    binary_dice,
    binary_precision,
    binary_recall,
    binary_f1,
    save_metric_values,
    create_overlay_image,
    count_parameters,
    estimate_flops,
)


#config
ROOT = Path("..").resolve()
CKPT = ROOT / "result_maskrcnn" / "checkpoint" / "last" / "last.pt"
OUT  = ROOT / "result_maskrcnn" / "eval"

SCORE_THRESH = 0.5
BATCH_SIZE = 1
NUM_WORKERS = 2

def get_transforms():
    return T.Compose([T.ToTensor()])


def combine_masks(masks: torch.Tensor, scores: torch.Tensor, thresh: float):
    keep = scores >= thresh
    if keep.sum() == 0:
        return None
    return (masks[keep] > 0.5).any(dim=0).to(torch.uint8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ensure_dir(OUT)
    overlay_dir = OUT / "overlay"
    ensure_dir(overlay_dir)

    
    _, val_loader, num_classes = get_dataloaders(
        root=ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_transforms=get_transforms(),
    )

    model = get_instance_segmentation_model(num_classes)
    model = load_model_checkpoint(model, CKPT, map_location=device)
    model.to(device)
    model.eval()

    dice_vals = []
    iou_vals  = []
    precision_vals = []
    recall_vals = []
    f1_vals = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            target = targets[0]

            output = model(images)[0]

            #  GT MASK (pindahkan ke CPU segera)
            gt_masks = target["masks"].cpu()
            if gt_masks.numel() == 0:
                _, H, W = images[0].shape
                gt_bin = torch.zeros((H, W), dtype=torch.uint8)
            else:
                gt_bin = (gt_masks > 0).any(dim=0).to(torch.uint8)

            #  PRED MASK 
            if len(output["masks"]) == 0:
                pred_bin = torch.zeros_like(gt_bin)
            else:
                pred_bin = combine_masks(
                    output["masks"].squeeze(1),
                    output["scores"],
                    SCORE_THRESH,
                )
                if pred_bin is None:
                    pred_bin = torch.zeros_like(gt_bin)

            gt_np   = gt_bin.numpy()
            pred_np = pred_bin.cpu().numpy() if pred_bin.is_cuda else pred_bin.numpy()

            dice = binary_dice(pred_np, gt_np)
            iou  = binary_iou(pred_np, gt_np)
            precision = binary_precision(pred_np, gt_np)
            recall = binary_recall(pred_np, gt_np)
            f1 = binary_f1(pred_np, gt_np)

            fname = target.get("file_name", "unknown")
            dice_vals.append((fname, dice))
            iou_vals.append((fname, iou))
            precision_vals.append((fname, precision))
            recall_vals.append((fname, recall))
            f1_vals.append((fname, f1))

            print(f"{fname} | Dice: {dice:.4f} | IoU: {iou:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

            #  OVERLAY 
            overlay = create_overlay_image(
                image=images[0].cpu(),
                boxes=output["boxes"].cpu(),
                labels=output["labels"].cpu(),
                masks=output["masks"].squeeze(1).cpu(),
                scores=output["scores"].cpu(),
                score_thresh=SCORE_THRESH,
            )

            overlay.save(overlay_dir / f"{Path(fname).stem}_overlay.png")
            
            # Bersihkan memori setiap batch
            del images, output, gt_masks, gt_bin, pred_bin, overlay
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()

    #  SAVE METRIC 
    save_metric_values(OUT / "dice.csv", dice_vals, "dice")
    save_metric_values(OUT / "iou.csv",  iou_vals,  "iou")
    save_metric_values(OUT / "precision.csv", precision_vals, "precision")
    save_metric_values(OUT / "recall.csv", recall_vals, "recall")
    save_metric_values(OUT / "f1.csv", f1_vals, "f1")

    mean_dice = sum(v for _, v in dice_vals) / max(len(dice_vals), 1)
    mean_iou  = sum(v for _, v in iou_vals)  / max(len(iou_vals), 1)
    mean_precision = sum(v for _, v in precision_vals) / max(len(precision_vals), 1)
    mean_recall = sum(v for _, v in recall_vals) / max(len(recall_vals), 1)
    mean_f1 = sum(v for _, v in f1_vals) / max(len(f1_vals), 1)

    # Hitung parameter dan FLOPs
    num_params = count_parameters(model)
    flops = estimate_flops(model)

    summary = {
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "num_images": len(dice_vals),
        "parameters": num_params,
        "parameters_M": num_params / 1e6,
        "FLOPs_G": flops if flops > 0 else "N/A",
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*60)
    print("EVALUASI SELESAI")
    print("="*60)
    print(f"Mean Dice:      {mean_dice:.4f}")
    print(f"Mean IoU:       {mean_iou:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall:    {mean_recall:.4f}")
    print(f"Mean F1:        {mean_f1:.4f}")
    print(f"Parameters:     {num_params:,} ({num_params/1e6:.2f}M)")
    if flops > 0:
        print(f"FLOPs:          {flops:.2f}G")
    print(f"\nHasil disimpan di: {OUT}")
    print("="*60)


if __name__ == "__main__":
    main()
