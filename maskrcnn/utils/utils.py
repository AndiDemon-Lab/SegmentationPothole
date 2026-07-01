import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    out_dir: Path,
    is_best: bool = False,
):
    ensure_dir(out_dir)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    torch.save(state, out_dir / "last.pt")
    if is_best:
        best_dir = out_dir.parent / "best"
        ensure_dir(best_dir)
        torch.save(state, best_dir / "best.pt")
    
    # Clear cache after checkpoint save
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_train_log(csv_path: Path, epoch: int, train_loss: float, lr: float, extra: Dict[str, Any] = None):
    ensure_dir(csv_path.parent)
    file_exists = csv_path.exists()
    fieldnames = ["epoch", "train_loss", "lr"]
    if extra:
        fieldnames.extend(list(extra.keys()))
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {"epoch": epoch, "train_loss": train_loss, "lr": lr}
        if extra:
            row.update(extra)
        writer.writerow(row)


def binary_iou(mask_pred: np.ndarray, mask_true: np.ndarray) -> float:
    mask_pred = mask_pred > 0
    mask_true = mask_true > 0
    intersection = np.logical_and(mask_pred, mask_true).sum()
    union = np.logical_or(mask_pred, mask_true).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def binary_dice(mask_pred: np.ndarray, mask_true: np.ndarray) -> float:
    mask_pred = mask_pred > 0
    mask_true = mask_true > 0
    intersection = np.logical_and(mask_pred, mask_true).sum()
    size_sum = mask_pred.sum() + mask_true.sum()
    if size_sum == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / size_sum


def binary_precision(mask_pred: np.ndarray, mask_true: np.ndarray) -> float:
    """Calculate precision: TP / (TP + FP)"""
    mask_pred = mask_pred > 0
    mask_true = mask_true > 0
    true_positive = np.logical_and(mask_pred, mask_true).sum()
    predicted_positive = mask_pred.sum()
    if predicted_positive == 0:
        return 1.0 if true_positive == 0 else 0.0
    return true_positive / predicted_positive


def binary_recall(mask_pred: np.ndarray, mask_true: np.ndarray) -> float:
    """Calculate recall: TP / (TP + FN)"""
    mask_pred = mask_pred > 0
    mask_true = mask_true > 0
    true_positive = np.logical_and(mask_pred, mask_true).sum()
    actual_positive = mask_true.sum()
    if actual_positive == 0:
        return 1.0 if true_positive == 0 else 0.0
    return true_positive / actual_positive


def binary_f1(mask_pred: np.ndarray, mask_true: np.ndarray) -> float:
    """Calculate F1 score: 2 * (precision * recall) / (precision + recall)"""
    precision = binary_precision(mask_pred, mask_true)
    recall = binary_recall(mask_pred, mask_true)
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: torch.nn.Module, input_size: Tuple[int, int, int] = (3, 512, 512)) -> float:
    """Estimate FLOPs for the model. Returns FLOPs in billions (GFLOPs)"""
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_size).to(device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9  # Convert to GFLOPs
    except ImportError:
        print("Warning: 'thop' package not found. Install with: pip install thop")
        return -1.0
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs: {e}")
        return -1.0


def save_metric_values(csv_path: Path, values: List[Tuple[str, float]], value_name: str):
    ensure_dir(csv_path.parent)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", value_name])
        for fname, val in values:
            writer.writerow([fname, val])


def plot_train_loss(csv_path: Path, out_path: Path):
    ensure_dir(out_path.parent)
    epochs = []
    losses = []
    if not csv_path.exists():
        return
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))
    if len(epochs) == 0:
        return
    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def create_overlay_image(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    class_map: Optional[Dict[int, str]] = None,
    score_thresh: float = 0.5,
) -> Image.Image:

    if image.ndim == 4:
        image = image[0]

    img = image.detach().cpu()
    if img.dtype.is_floating_point:
        img = (img.clamp(0, 1) * 255).to(torch.uint8)

    img_np = img.permute(1, 2, 0).numpy()
    base = Image.fromarray(img_np)

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    boxes = boxes.detach().cpu()
    labels = labels.detach().cpu()
    masks = masks.detach().cpu()
    if scores is not None:
        scores = scores.detach().cpu()

    n = masks.shape[0]
    if n == 0:
        return base

    if scores is not None:
        keep = scores >= score_thresh
        if keep.sum() == 0:
            return base
        boxes = boxes[keep]
        labels = labels[keep]
        masks = masks[keep]
        scores = scores[keep]

    H, W = masks.shape[1], masks.shape[2]
    overlay_np = np.zeros((H, W, 4), dtype=np.uint8)

    colors = [
        (255, 0, 0, 80),
        (0, 255, 0, 80),
        (0, 0, 255, 80),
        (255, 255, 0, 80),
        (255, 0, 255, 80),
        (0, 255, 255, 80),
    ]

    for i in range(masks.shape[0]):
        mask = masks[i].numpy() > 0.5
        overlay_np[mask] = colors[i % len(colors)]

    overlay = Image.fromarray(overlay_np, mode="RGBA")
    out = base.convert("RGBA")
    out = Image.alpha_composite(out, overlay)

    draw = ImageDraw.Draw(out)
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].tolist()
        cls_id = int(labels[i].item())
        name = class_map.get(cls_id, str(cls_id)) if class_map else str(cls_id)
        txt = name
        if scores is not None:
            txt = f"{name} {scores[i].item():.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=2)
        text_bg = [x1, y1 - 12, x1 + 8 * len(txt), y1]
        draw.rectangle(text_bg, fill=(0, 0, 0, 160))
        draw.text((x1 + 2, y1 - 12), txt, fill=(255, 255, 255, 255))

    return out.convert("RGB")
