from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T

from dataloader.getdata import get_dataloaders
from model.maskrcnn import get_instance_segmentation_model
from utils.utils import (
    set_seed,
    ensure_dir,
    save_checkpoint,
    save_train_log,
    plot_train_loss,
)


#config
ROOT = Path("..").resolve()
OUT  = ROOT / "result_maskrcnn"

EPOCHS = 100  
BATCH_SIZE = 1
LR = 5e-5 
WARMUP_EPOCHS = 3
GRAD_CLIP = 1.0  
NUM_WORKERS = 2  
SEED = 42

set_seed(SEED)
ensure_dir(OUT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


transform = T.ToTensor()

train_loader, test_loader, num_classes = get_dataloaders(
    root=ROOT,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    image_transforms=transform,
)


#model
model = get_instance_segmentation_model(num_classes)
model.to(device)

optimizer = AdamW(
    model.parameters(), 
    lr=LR,
    weight_decay=1e-4,  # L2 regularization
)

# Cosine annealing scheduler untuk gradual LR decay
scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=EPOCHS - WARMUP_EPOCHS,
    eta_min=1e-6
)


#train
best_loss = float("inf")
log_csv = OUT / "train_log.csv"

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    print(f"\n=== Epoch {epoch}/{EPOCHS} ===")

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        
        # Check for NaN/Inf loss
        if not torch.isfinite(loss):
            print(f"⚠️ Warning: Non-finite loss detected at batch {batch_idx}. Skipping...")
            continue

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping untuk stabilitas
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()

        total_loss += loss.item()
        
        # Print detail loss setiap 20 batch untuk monitoring
        if (batch_idx + 1) % 20 == 0:
            loss_str = " | ".join([f"{k}: {v.item():.3f}" for k, v in loss_dict.items()])
            print(f"  Batch {batch_idx+1}: Total={loss.item():.4f} | {loss_str}")
        
        # Bersihkan cache VRAM setiap 10 batch
        del loss, loss_dict
        if batch_idx % 10 == 0:  
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    
    # Learning rate warmup
    if epoch <= WARMUP_EPOCHS:
        warmup_lr = LR * (epoch / WARMUP_EPOCHS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
    else:
        # Scheduler step setelah warmup
        scheduler.step()
    
    lr_now = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | LR: {lr_now:.6f}")

    #log c
    save_train_log(
        csv_path=log_csv,
        epoch=epoch,
        train_loss=avg_loss,
        lr=lr_now,
    )

    #cekpoint
    is_best = avg_loss < best_loss
    if is_best:
        best_loss = avg_loss

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_metric=best_loss,
        out_dir=OUT / "checkpoint" / "last",
        is_best=is_best,
    )

    #loss plot
    plot_train_loss(
        csv_path=log_csv,
        out_path=OUT / "loss.png"
    )

print("\n" + "="*60)
print("TRAINING SELESAI")
print("="*60)
print(f"Best loss: {best_loss:.4f}")
print(f"Checkpoint: {OUT / 'checkpoint' / 'last' / 'last.pt'}")
if best_loss < 0.3:
    print("✓ Model bagus! Siap untuk evaluasi")
elif best_loss < 0.5:
    print("⚠️ Model cukup baik, bisa dicoba evaluasi")
else:
    print("⚠️ Loss masih tinggi, pertimbangkan train lebih lama")
print("="*60)
