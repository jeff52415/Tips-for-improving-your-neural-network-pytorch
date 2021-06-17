import torch
from torchvision import models

from src.no_bias_decay import add_weight_decay


def test_decay():
    model = models.resnet18()
    params = add_weight_decay(model, 2e-5)
    torch.optim.SGD(params, lr=0.001)
