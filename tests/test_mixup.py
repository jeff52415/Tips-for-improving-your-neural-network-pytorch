import torch

from src.mixup import Mixup


def test_mixup():
    from src.efficientnet_lite import build_efficientnet_lite

    mixup = Mixup()

    model_name = "efficientnet_lite3"
    model = build_efficientnet_lite(
        model_name,
        num_classes=10,
        group_normalization=False,
        weight_standardization=False,
        stochastic_depth=False,
    )

    images = torch.randn(2, 3, 32, 32)
    labels = torch.randint(1, 10, (2,))
    mixup(model, images, labels)
