from pathlib import Path
from torch.utils.data import DataLoader

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
