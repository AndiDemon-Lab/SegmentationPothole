#VISUALISASI HASIL PREDIKSI

from pathlib import Path
import argparse
import math
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

# import package / fallback
try:
    from .models.unet_model import UNet
    from .data.dataset import PotholeDataset
    from .utils.utils import ensure_dir
except:
    from models.unet_model import UNet
    from data.dataset import PotholeDataset
    from utils.utils import ensure_dir


# ---- Konversi tensor → PIL ----
def tensor_to_pil(t):
    arr = t.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1,3):
        arr = arr.transpose(1,2,0)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.dtype in (np.float32, np.float64):
        arr = (np.clip(arr,0,1)*255).astype(np.uint8)
    return Image.fromarray(arr)


def mask_to_pil(mask, thr=0.5):
    m = mask.detach().cpu().numpy()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    m = (m > thr).astype(np.uint8)*255
    return Image.fromarray(m, mode="L")


# ---- Overlay ----
def overlay(img, mask, color=(0,0,255), alpha=0.35):
    img = img.convert("RGBA")
    m = mask.resize(img.size, Image.NEAREST)
    m_arr = np.array(m)

    # buat layer RGBA
    ov = np.zeros((img.size[1], img.size[0], 4), dtype=np.uint8)
    ov[...,0] = color[0]
    ov[...,1] = color[1]
    ov[...,2] = color[2]
    ov[...,3] = (m_arr/255 * (255*alpha)).astype(np.uint8)

    ov_img = Image.fromarray(ov, "RGBA")
    return Image.alpha_composite(img, ov_img).convert("RGB")


# ---- Kolase ----
def make_collage(imgs, cols, tile):
    """Gabungkan list gambar menjadi kolase."""
    rows = math.ceil(len(imgs)/cols)
    W, H = cols*tile, rows*tile
    canvas = Image.new("RGB",(W,H),(40,40,40))
    for i,im in enumerate(imgs):
        r, c = divmod(i, cols)
        im_r = im.resize((tile,tile))
        canvas.paste(im_r,(c*tile, r*tile))
    return canvas


# ---- Argumen ----
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["last","best"], default="last")
    ap.add_argument("--out-dir", type=str, default="result_train")
    ap.add_argument("--list-dir", type=str, default="dataset_split")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--collage", action="store_true", default=True)
    ap.add_argument("--collage-cols", type=int, default=5)
    ap.add_argument("--collage-tile", type=int, default=256)
    return ap.parse_args()


# ---- Cari checkpoint ----
def find_ckpt(ckpt_dir, mode):
    if mode == "last":
        p = ckpt_dir/"unet_last.pt"
        if p.exists(): return p
        p2 = ckpt_dir/"unet_best.pt"
        if p2.exists(): return p2
    else:
        p = ckpt_dir/"unet_best.pt"
        if p.exists(): return p
        p2 = ckpt_dir/"unet_last.pt"
        if p2.exists(): return p2
    raise SystemExit("Checkpoint tidak ditemukan.")


# ---- Main ----
def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(a.amp and device.type=="cuda")

    out_dir = Path(a.out_dir)
    ckpt_dir = out_dir/"checkpoints"
    vis_dir = out_dir/"vis"
    overlays_dir = vis_dir/"overlays"
    ensure_dir(vis_dir); ensure_dir(overlays_dir)

    ckpt = find_ckpt(ckpt_dir, a.mode)
    print(f"[viz] checkpoint: {ckpt.name}")

    # load model (sesuaikan base_c / bilinear jika berbeda)
    model = UNet(n_channels=3, n_classes=1, bilinear=True, base_c=64).to(device)
    state = torch.load(ckpt, map_location=device)
    state = state.get("model", state)
    model.load_state_dict(state, strict=False)
    model.eval()

    # dataset
    list_dir = Path(a.list_dir)
    test_list = list_dir/"test.txt"
    test_set = PotholeDataset(img_size=a.img_size, list_file=test_list)
    loader = DataLoader(test_set, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=(device.type=="cuda"))

    saved = []
    count = 0

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(imgs)
                preds = torch.sigmoid(logits)

            for i in range(imgs.size(0)):
                if count >= a.num_samples: break
                img_pil = tensor_to_pil(imgs[i].cpu())
                mask_pil = mask_to_pil(preds[i].cpu())

                ov = overlay(img_pil, mask_pil)
                ov.save(overlays_dir/f"overlay_{count:04d}.png")
                saved.append(ov)
                count += 1

            if count >= a.num_samples:
                break

    print(f"[viz] {count} overlay disimpan ke: {overlays_dir}")

    # kolase
    if a.collage and saved:
        col = make_collage(saved, cols=a.collage_cols, tile=a.collage_tile)
        col.save(vis_dir/"collage.png")
        print(f"[viz] collage disimpan ke: {vis_dir/'collage.png'}")

    print("[viz] selesai.")


if __name__ == "__main__":
    main()
