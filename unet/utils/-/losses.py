import torch
import torch.nn as nn

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
