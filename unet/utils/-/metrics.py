import torch

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
