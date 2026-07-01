import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from models.unet_model import UNet
from data.dataset import PotholeDataset
from utils.utils import (set_seed, 
                         ensure_dir, 
                          dice_from_logits, 
                          compute_pos_weight, 
                          BCEDiceLoss, 
                          TverskyLoss, 
                          save_history_csv, 
                          plot_training_curves)

BASE = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser(description="Train UNet pothole segmentation")
    ap.add_argument("--list-dir", type=str, default=str(BASE / "dataset_split"))
    ap.add_argument("--augment", choices=["on", "off"], default="off")
    ap.add_argument("--aug-images-dir", type=str, default=str(BASE / "augment_out/images"))
    ap.add_argument("--aug-masks-dir", type=str, default=str(BASE / "augment_out/masks"))
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2) #sesuaikan dengan VRAM
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-4) #0.0005
    ap.add_argument("--weight-decay", type=float, default=1e-5) #buat mencegah overfitting dengan regularisasi
    ap.add_argument("--workers", type=int, default=4) #menggunakan 4 worker untuk loading data
    ap.add_argument("--out-dir", type=str, default=str(BASE / "result_train"))
    ap.add_argument("--bilinear", action="store_true", default=True)
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--amp", action="store_true") #penghematan memori 
    ap.add_argument("--pos-weight", type=float, default=3.0) #bobot untuk pothole 
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    OUT = Path(args.out_dir)
    CKPT = OUT / "checkpoints"
    LOGS = OUT / "logs"
    for d in (CKPT, LOGS):
        ensure_dir(d)
    #muat data pelatihan
    list_dir = Path(args.list_dir)
    train_list = list_dir / "train.txt"

    train_base = PotholeDataset(img_size=args.img_size, list_file=train_list)

    if args.augment == "on":
        ds_aug = PotholeDataset(Path(args.aug_images_dir), Path(args.aug_masks_dir),
                                img_size=args.img_size)
        train_set = ConcatDataset([train_base, ds_aug])
    else:
        train_set = train_base

    if args.pos_weight is not None:
        pw = float(args.pos_weight)
    else:
        pw, _, _, _ = compute_pos_weight(
            train_set if args.augment == "on" else train_base,
            batch_size=args.batch_size,
            workers=args.workers
        )

    pos_w = torch.tensor([pw], device=device)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )
    #buat model UNet
    net = UNet(n_channels=3, n_classes=1,
               bilinear=args.bilinear, base_c=args.base_ch).to(device) #(bch, 3, H, W) -> (bch, 1, H, W)
    #buat optimizer, scheduler, loss function
    opt = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(opt, max_lr=args.lr, 
                           epochs=args.epochs,
                           steps_per_epoch=len(train_loader),
                           pct_start=0.1, # warm-up 10% dari total iterasi
                           div_factor=10, # awal lr = max_lr/div_factor 5e-5
                           final_div_factor=1000)

    loss_fn = BCEDiceLoss(bce_weight=0.5, pos_weight=pos_w)
    tversky_fn = TverskyLoss(alpha=0.5, beta=0.5) #sama dengan dice loss
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    best_loss = float("inf")

    history = {"epoch": [], "train_loss": [], "train_dice": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        net.train()
        total_loss = 0.0
        total_dice = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = net(imgs)
                loss = 0.5 * loss_fn(logits, masks) + 0.5 * tversky_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            b = imgs.size(0)
            total_loss += float(loss.item()) * b
            total_dice += dice_from_logits(logits, masks) * b
            total_samples += b

            pbar.set_postfix(loss=float(loss.item()))

        avg_loss = total_loss / total_samples
        avg_dice = total_dice / total_samples

        lr = scheduler.get_last_lr()[0]

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["train_dice"].append(avg_dice)
        history["lr"].append(lr)

        print(f"[epoch {epoch}] loss={avg_loss:.4f} dice={avg_dice:.4f} lr={lr:.6f}")

        ckpt_payload = {
            "model": net.state_dict(),
            "epoch": epoch,
            "img_size": args.img_size,
            "base_ch": args.base_ch,
            "bilinear": args.bilinear
        }

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_payload, CKPT / "unet_best.pt")

        torch.save(ckpt_payload, CKPT / "unet_last.pt")

        save_history_csv(LOGS, history)
        plot_training_curves(LOGS, history, tag="train")

    save_history_csv(LOGS, history)
    plot_training_curves(LOGS, history, tag="train")

    print("[train] selesai")


if __name__ == "__main__":
    main()
    