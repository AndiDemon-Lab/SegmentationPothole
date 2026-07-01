import torch
from thop import profile, clever_format

# IMPORT UNet (RELATIVE IMPORT)
from .unet_model import UNet


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    model = UNet(
        n_channels=3,
        n_classes=1,
        bilinear=True,
        base_c=64
    )
    model.eval()

    # ===== PARAMS =====
    total, trainable = count_params(model)
    print(f"Total Params     : {total:,}")
    print(f"Trainable Params : {trainable:,}")

    # ===== FLOPs =====
    dummy_input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs  : {flops}")
    print(f"Params : {params}")


if __name__ == "__main__":
    main()
