import torch
from thop import profile, clever_format
from typing import Optional

from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_instance_segmentation_model(
    num_classes: int,
    pretrained_backbone: bool = True,
    min_size: int = 400,
    max_size: int = 640,
) -> MaskRCNN:

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights="DEFAULT" if pretrained_backbone else None,
        trainable_layers=3,
    )

    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
    )

    return model


def main():
    device = "cpu"

    # ==========================
    # Build Mask R-CNN
    # ==========================
    model = get_instance_segmentation_model(
        num_classes=2,   # background + pothole
        pretrained_backbone=True
    )
    model.eval().to(device)

    # ==========================
    # Parameter count (FULL MODEL)
    # ==========================
    total, trainable = count_params(model)
    print("=== PARAMETER COUNT (Mask R-CNN) ===")
    print(f"Total Params     : {total:,}")
    print(f"Trainable Params : {trainable:,}")

    # ==========================
    # FLOPs (BACKBONE + FPN ONLY)
    # ==========================
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    backbone = model.backbone
    backbone.eval()

    flops, params = profile(
        backbone,
        inputs=(dummy_input,),
        verbose=False
    )

    flops, params = clever_format([flops, params], "%.3f")

    print("\n=== FLOPs (Backbone + FPN) ===")
    print(f"Input Size : 512 x 512")
    print(f"FLOPs      : {flops}")
    print(f"Params     : {params}")


if __name__ == "__main__":
    main()
