import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PotholeDataset(Dataset):
    def __init__(self, list_file, img_size=512):
        self.samples = []
        with open(list_file, "r") as f:
            for line in f:
                img, mask = line.strip().split()
                self.samples.append((img, mask))
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        mask = np.array(mask, dtype=np.uint8)

        # ===== INI BARIS KRITIS =====
        mask = (mask > 0).astype(np.int64)

        mask = torch.from_numpy(mask)

        return image, mask
