import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np


class PotholeCocoDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split_file: Path,
        annotation_file: Path,
        transforms=None,
    ):

        self.root = Path(root)
        self.split_file = Path(split_file)
        self.annotation_file = Path(annotation_file)
        self.transforms = transforms

        # load list image untuk split ini
        with open(self.split_file, "r") as f:
            self.image_names = [line.strip() for line in f if line.strip()]

        # load anotasi COCO
        with open(self.annotation_file, "r") as f:
            coco = json.load(f)

        # map image_id -> info gambar
        self.images_info: Dict[int, Dict[str, Any]] = {
            img["id"]: img for img in coco["images"]
        }

        # map file_name -> image_id
        self.filename_to_id: Dict[str, int] = {
            img["file_name"]: img["id"] for img in coco["images"]
        }

        # kumpulkan anotasi per image_id
        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            self.annotations_by_image.setdefault(img_id, []).append(ann)

        # categories (kalau cuma 1 class pothole: background(0) + pothole(1))
        self.categories = coco.get("categories", [])
        self.num_classes = len(self.categories) + 1  # + background

    def __len__(self) -> int:
        return len(self.image_names)

    def _polygons_to_mask(
        self, polygons: List[List[float]], height: int, width: int
    ) -> np.ndarray:
        """
        Convert list of polygon (COCO 'segmentation') ke binary mask (H, W).
        """
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for poly in polygons:
            # poly = [x1, y1, x2, y2, ...]
            xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(xy, outline=1, fill=1)

        return np.array(mask, dtype=np.uint8)

    def __getitem__(self, idx: int):
        file_name = self.image_names[idx]
        img_id = self.filename_to_id[file_name]
        img_info = self.images_info[img_id]

        img_path = self.root / "images" / file_name
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        anns = self.annotations_by_image.get(img_id, [])

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        attributes_list = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])  # sudah diasumsikan mulai dari 1
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

            seg = ann.get("segmentation", [])
            # segmentation bisa beberapa polygon
            if len(seg) > 0:
                mask = self._polygons_to_mask(seg, height, width)
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
            masks.append(mask)

            attributes_list.append(ann.get("attributes", {}))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
            "attributes": attributes_list,
            "file_name": file_name,
        }

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


def collate_fn(batch):
    """
    Collate function standar untuk detection (list of image, list of target).
    """
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def get_dataloaders(
    root: str,
    annotation_file: str = "annotation/dataset_meta.json",
    train_split: str = "dataset_split/trainmaskrcnn.txt",
    test_split: str = "dataset_split/testmaskrcnn.txt",
    batch_size: int = 2,
    num_workers: int = 4,
    image_transforms=None,
) -> Tuple[DataLoader, DataLoader, int]:
    root = Path(root)
    annotation_path = root / annotation_file
    train_split_path = root / train_split
    test_split_path = root / test_split

    train_dataset = PotholeCocoDataset(
        root=root,
        split_file=train_split_path,
        annotation_file=annotation_path,
        transforms=image_transforms,
    )

    test_dataset = PotholeCocoDataset(
        root=root,
        split_file=test_split_path,
        annotation_file=annotation_path,
        transforms=image_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # evaluasi per gambar
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    num_classes = train_dataset.num_classes

    return train_loader, test_loader, num_classes
