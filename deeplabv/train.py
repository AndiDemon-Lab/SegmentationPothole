import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from dataloader.getData import PotholeDataset
from model.deeplab import DeepLabV3
from utils.utils import plot_loss

torch.backends.cudnn.benchmark = True

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "resultdeeplabv"
OUT.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"

train_ds = PotholeDataset(
    BASE / "dataset_split/train.txt",
    img_size=512
)

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

model = DeepLabV3(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

epochs = 100
losses = []

epoch_bar = tqdm(
    range(1, epochs + 1),
    desc="Training",
    ncols=60,
    dynamic_ncols=False,
    ascii=True
)

for epoch in epoch_bar:
    model.train()
    total_loss = 0.0

    for img, mask in train_loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(img)
            loss = criterion(out, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)

    tqdm.write(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), OUT / "model.pt")
plot_loss(losses, OUT / "loss_plot.png")

with open(OUT / "train_log.txt", "w") as f:
    for i, l in enumerate(losses):
        f.write(f"{i+1},{l:.6f}\n")
