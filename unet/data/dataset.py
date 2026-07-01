from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

ALLOWED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

#kumpulkan gambar 
def _collect(dirp: Path) -> List[Path]:
    return [p for p in sorted(dirp.rglob("*")) if p.suffix.lower() in ALLOWED_EXTS]

def _map_by_stem(paths: List[Path]):
    return {p.stem: p for p in paths}

#baca pasangan dari file list
def read_pairs_from_list(list_file: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    list_file = Path(list_file).resolve()
    # base = parent dari folder dataset_split → biasanya project root
    base = list_file.parent.parent
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split("\t")
            pa = Path(a)
            pb = Path(b)
            if not pa.is_absolute():
                pa = (base / a).resolve()
            if not pb.is_absolute():
                pb = (base / b).resolve()
            pairs.append((pa, pb))
    return pairs

#Dataset class
class PotholeDataset(Dataset):
    def __init__(
        self,
        images_dir: Optional[Path] = None,
        masks_dir: Optional[Path] = None,
        img_size: int = 512,
        list_file: Optional[Path] = None,
        pairs: Optional[List[Tuple[Path, Path]]] = None,
    ):
        self.img_size = img_size # ukuran gambar input

        if list_file is not None:
            pairs = read_pairs_from_list(Path(list_file))
        elif pairs is None:
            assert images_dir is not None and masks_dir is not None, \
                "Berikan images_dir+masks_dir atau list_file/pairs"
            images_dir = Path(images_dir); masks_dir = Path(masks_dir)
            imgs  = _collect(images_dir)
            masks = _collect(masks_dir)
            mask_map = _map_by_stem(masks)

            pairs = []
            missing = []
            for img in imgs:
                m = mask_map.get(img.stem)
                if m: pairs.append((img, m))
                else: missing.append(img.name)

            if not pairs:
                raise RuntimeError("Tidak ditemukan pasangan image-mask. Samakan nama.")
            if missing:
                print(f"[dataset] Warning: {len(missing)} image tanpa mask. Contoh: {missing[:5]}")

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    #membuka gambar dengan koreksi orientasi EXIF
    def _load_exif_safe(p: Path) -> Image.Image:
        return ImageOps.exif_transpose(Image.open(p))

    def __getitem__(self, i: int):
        #load image dan mask
        img_path, mask_path = self.pairs[i]
        img  = self._load_exif_safe(img_path).convert("RGB")
        mask = self._load_exif_safe(mask_path)
        if mask.mode != "L":
            mask = mask.convert("L")

        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        #buat tensor
        img_t  = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        mask_t = (mask_t > 0.5).float()
        return img_t, mask_t
