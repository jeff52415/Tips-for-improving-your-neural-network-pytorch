import torch

from src.efficientnet_lite import build_efficientnet_lite


def test_bn():
    model_name = "efficientnet_lite3"
    model = build_efficientnet_lite(
        model_name,
        num_classes=10,
        group_normalization=False,
        weight_standardization=False,
        stochastic_depth=False,
    )
    input_ = torch.randn(1, 3, 64, 64)
    model.features(input_)


def test_gn():
    model_name = "efficientnet_lite3"
    model = build_efficientnet_lite(
        model_name,
        num_classes=10,
        group_normalization=True,
        weight_standardization=False,
        stochastic_depth=False,
    )
    input_ = torch.randn(1, 3, 64, 64)
    model.features(input_)


def test_ws():
    model_name = "efficientnet_lite3"
    model = build_efficientnet_lite(
        model_name,
        num_classes=10,
        group_normalization=True,
        weight_standardization=True,
        stochastic_depth=False,
    )
    input_ = torch.randn(1, 3, 64, 64)
    model.features(input_)


def test_sd():
    model_name = "efficientnet_lite3"
    model = build_efficientnet_lite(
        model_name,
        num_classes=10,
        group_normalization=True,
        weight_standardization=True,
        stochastic_depth=True,
    )
    input_ = torch.randn(1, 3, 64, 64)
    model.features(input_)


def test_full():
    model_name = "efficientnet_lite3"
    model = build_efficientnet_lite(
        model_name,
        num_classes=10,
        group_normalization=False,
        weight_standardization=False,
        stochastic_depth=False,
    )
    input_ = torch.randn(1, 3, 64, 64)
    output = model(input_)
    assert output.shape[1] == 10
