from typing import Optional

import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_instance_segmentation_model(
    num_classes: int,
    pretrained_backbone: bool = True,
    min_size: int = 800,   # Naikkan untuk image besar
    max_size: int = 1333,  # Naikkan untuk image besar
    # min_size: int = 512,   # Kurangi untuk memory
    # max_size: int = 800,   # Kurangi untuk memory
    trainable_layers: int = 5,  # Train lebih banyak layer
) -> MaskRCNN:

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights="DEFAULT" if pretrained_backbone else None,
        trainable_layers=trainable_layers,  # Default 5 agar lebih fleksibel
    )

    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        min_size=min_size,   
        max_size=max_size,
        # Threshold untuk RPN dan ROI
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=1000,
    )

    return model


def load_model_checkpoint(
    model: MaskRCNN,
    checkpoint_path: str,
    map_location: Optional[str] = "cpu",
) -> MaskRCNN:

    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state["model"])
    return model
