from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv
import random
import numpy as np
import torch
import torch.nn as nn

#set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# Loss functions
def _ensure_shapes_loss(logits, targets):
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.type_as(logits)
    return logits, targets

def dice_loss(logits, targets, eps=1e-6):
    logits, targets = _ensure_shapes_loss(logits, targets)
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return (1.0 - (num / den)).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, pos_weight: torch.Tensor = None):
        super().__init__()
        self.w = bce_weight
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        logits, targets = _ensure_shapes_loss(logits, targets)
        bce_val = self.bce(logits, targets)
        dice_val = dice_loss(logits, targets)
        return self.w * bce_val + (1 - self.w) * dice_val

class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        logits, targets = _ensure_shapes_loss(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        tp = (probs * targets).sum(dim=dims)
        fn = ((1 - probs) * targets).sum(dim=dims)
        fp = (probs * (1 - targets)).sum(dim=dims)
        tversky_index = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        loss = 1.0 - tversky_index
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

# metrics
def _ensure_shapes(logits, targets):
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.type_as(logits)
    return logits, targets


def dice_from_logits(logits, targets, eps=1e-6):
    logits, targets = _ensure_shapes(logits, targets)
    p = torch.sigmoid(logits)
    p = p.view(p.size(0), -1)
    t = targets.view(targets.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def dice_coef(logits, targets, thr=0.5, eps=1e-6):
    logits, targets = _ensure_shapes(logits, targets)
    p = torch.sigmoid(logits)
    pred = (p > thr).float()
    num = 2 * (pred * targets).sum(dim=(2, 3)) + eps
    den = (pred + targets).sum(dim=(2, 3)) + eps
    return (num / den).mean()


def iou_coef(logits, targets, thr=0.5, eps=1e-6):
    logits, targets = _ensure_shapes(logits, targets)
    p = torch.sigmoid(logits)
    pred = (p > thr).float()
    inter = (pred * targets).sum(dim=(2, 3)) + eps
    union = (pred + targets - pred * targets).sum(dim=(2, 3)) + eps
    return (inter / union).mean()

#plotting functions
def save_history_csv(out_dir: Path, history: dict, filename: str = "history.csv"):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    lengths = [len(history.get(k, [])) for k in keys]
    max_len = max(lengths) if lengths else 0
    rows = []
    for i in range(max_len):
        row = []
        for k in keys:
            vals = history.get(k, [])
            row.append(vals[i] if i < len(vals) else "")
        rows.append(row)

    csv_path = out_dir / filename
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerows(rows)

def _plot_line(xs, ys, title, xlabel, ylabel, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

# Buat kurva training/validation berdasarkan history dict
def plot_training_curves(out_dir: Path, history: dict, tag: str = "val"):
    epochs = history.get("epoch", None)
    if epochs is None:
        n = len(history.get("train_loss", []))
        epochs = list(range(1, n + 1))

    out_dir.mkdir(parents=True, exist_ok=True)

    if "train_loss" in history:
        _plot_line(epochs, history["train_loss"], "Training Loss", "Epoch", "Loss",
                   out_dir / "curve_train_loss.png")

    if f"{tag}_dice" in history:
        _plot_line(epochs, history[f"{tag}_dice"], f"{tag.upper()} Dice", "Epoch", "Dice",
                   out_dir / f"curve_{tag}_dice.png")
    if f"{tag}_iou" in history:
        _plot_line(epochs, history[f"{tag}_iou"], f"{tag.upper()} IoU", "Epoch", "IoU",
                   out_dir / f"curve_{tag}_iou.png")
    if "lr" in history:
        _plot_line(epochs, history["lr"], "Learning Rate", "Epoch", "LR",
                   out_dir / "curve_lr.png")

#pos weight calculation for imbalanced datasets
def compute_pos_weight(dataset, batch_size=8, workers=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=False)

    total_pos = 0.0
    total_pixels = 0

    for _, masks in loader:
        if masks.dim() == 4:
            m = masks.view(masks.size(0), -1)
        elif masks.dim() == 3:
            m = masks.view(m.size(0), -1)
        else:
            m = masks.view(-1)

        total_pos += float(m.sum().item())
        total_pixels += int(m.numel())

    total_neg = total_pixels - total_pos
    total_pos = max(total_pos, 1.0)
    total_neg = max(total_neg, 1.0)

    pw = total_neg / total_pos
    pw = float(max(1.0, min(pw, 100.0)))

    return pw, int(total_pos), int(total_neg), int(total_pixels)
