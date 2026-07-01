import argparse
from pathlib import Path
import json
import torch
import torchvision.transforms as T

from dataloader.getdata import get_dataloaders
from model.maskrcnn import get_instance_segmentation_model, load_model_checkpoint
from utils.utils import ensure_dir


def get_transforms():
    return T.Compose([T.ToTensor()])


def auto_find_last_checkpoint(root: Path) -> Path:
    ckpt = root / "resultmaskrcnn" / "checkpoint" / "last" / "last.pth"
    if ckpt.exists():
        return ckpt
    raise FileNotFoundError("Checkpoint Mask R-CNN tidak ditemukan.")


def combine_masks(masks: torch.Tensor, scores: torch.Tensor, score_thresh: float):
    keep = scores >= score_thresh
    if keep.sum() == 0:
        return None
    masks = masks[keep]
    return (masks > 0.5).any(dim=0).to(torch.uint8)


def eval_prf(
    model,
    dataloader,
    device,
    score_thresh: float = 0.5,
):
    model.eval()
    tp = fp = fn = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            target = targets[0]
            output = outputs[0]

            gt_masks = target["masks"].to(device)

            if gt_masks.numel() == 0:
                _, H, W = images[0].shape
                gt_binary = torch.zeros((H, W), dtype=torch.uint8, device=device)
            else:
                gt_binary = (gt_masks > 0).any(dim=0).to(torch.uint8)

            if len(output["masks"]) == 0:
                pred_binary = torch.zeros_like(gt_binary)
            else:
                pred_binary = combine_masks(
                    masks=output["masks"].squeeze(1),
                    scores=output["scores"],
                    score_thresh=score_thresh,
                )
                if pred_binary is None:
                    pred_binary = torch.zeros_like(gt_binary)

            pred = pred_binary.float()
            gt = gt_binary.float()

            tp += torch.sum(pred * gt).item()
            fp += torch.sum(pred * (1 - gt)).item()
            fn += torch.sum((1 - pred) * gt).item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mask R-CNN (Precision, Recall, F1)")
    parser.add_argument("--root", type=str, default="..")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="resultmaskrcnn/eval_prf")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    ensure_dir(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, num_classes = get_dataloaders(
        root=root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_transforms=get_transforms(),
    )

    checkpoint_path = auto_find_last_checkpoint(root)

    model = get_instance_segmentation_model(num_classes=num_classes)
    model = load_model_checkpoint(model, str(checkpoint_path), map_location=device)
    model.to(device)

    precision, recall, f1 = eval_prf(
        model=model,
        dataloader=val_loader,
        device=device,
        score_thresh=args.score_thresh,
    )

    print(f"[Mask R-CNN] Precision: {precision:.4f}")
    print(f"[Mask R-CNN] Recall   : {recall:.4f}")
    print(f"[Mask R-CNN] F1-score : {f1:.4f}")

    with open(output_dir / "prf_summary.json", "w") as f:
        json.dump(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            f,
            indent=2,
        )

    print(f"[eval] PRF metrics disimpan di {output_dir}")


if __name__ == "__main__":
    main()
