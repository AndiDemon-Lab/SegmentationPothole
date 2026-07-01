import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from pathlib import Path
from thop import profile

from dataloader.getData import PotholeDataset
from model.deeplab import DeepLabV3
from utils.utils import dice_score, iou_score, precision_recall_f1

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "resultdeeplabv"
OUT.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

img_size = 512

test_ds = PotholeDataset(
    BASE / "dataset_split/test.txt",
    img_size=img_size
)
loader = DataLoader(test_ds, batch_size=1, shuffle=False)

model = DeepLabV3(num_classes=2).to(device)
model.load_state_dict(torch.load(OUT / "model.pt", map_location=device))
model.eval()

# ================= PARAM & FLOPs =================
dummy = torch.randn(1, 3, img_size, img_size).to(device)
flops, params = profile(model, inputs=(dummy,), verbose=False)

params_m = params / 1e6
flops_g = flops / 1e9

print("==== Model Complexity ====")
print(f"Parameters : {params_m:.2f} M")
print(f"FLOPs      : {flops_g:.2f} G")

# ================= EVALUATION =================
dice_all, iou_all, p_all, r_all, f_all = [], [], [], [], []

overlay_images = []
max_overlay = 10

with torch.no_grad():
    for idx, (img, mask) in enumerate(loader):
        img = img.to(device)
        pred = model(img)
        pred = torch.argmax(pred, dim=1).cpu().numpy()[0]
        gt = mask.numpy()[0]

        dice_all.append(dice_score(pred, gt))
        iou_all.append(iou_score(pred, gt))

        p, r, f = precision_recall_f1(
            pred.flatten(), gt.flatten()
        )
        p_all.append(p)
        r_all.append(r)
        f_all.append(f)

        # ================= OVERLAY =================
        if idx < max_overlay:
            img_np = img.cpu().numpy()[0].transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)

            overlay = img_np.copy()

            overlay[gt == 1] = [0, 255, 0]      # GT = hijau
            overlay[pred == 1] = [255, 0, 0]   # Pred = merah

            overlay = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
            overlay_images.append(overlay)

# ================= COLLAGE =================
if len(overlay_images) > 0:
    rows = 2
    cols = 5
    h, w, _ = overlay_images[0].shape
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for i, im in enumerate(overlay_images):
        r = i // cols
        c = i % cols
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = im

    cv2.imwrite(str(OUT / "overlay_collage.png"), canvas)

# ================= SUMMARY =================
summary = {
    "Dice": np.mean(dice_all),
    "IoU": np.mean(iou_all),
    "Precision": np.mean(p_all),
    "Recall": np.mean(r_all),
    "F1-score": np.mean(f_all),
    "Params(M)": params_m,
    "FLOPs(G)": flops_g
}

with open(OUT / "metric_summary.txt", "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v:.4f}\n")

print("==== Evaluation Result ====")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")
