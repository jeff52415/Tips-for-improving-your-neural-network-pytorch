from typing import Union

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def build_loader(batch_size: int = 32, num_workers: int = 2):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, pin_memory=True, num_workers=num_workers
    )
    return trainloader, testloader


def accuracy_score(
    prediction: Union[np.ndarray, list], target: Union[np.ndarray, list]
):
    count = 0
    length = len(prediction)

    if len(prediction) != len(target):
        raise ValueError(
            f"Mismatch between prediction and target, got {len(prediction)} items in prediction & {len(target)} items in target"
        )
    for p, t in zip(prediction, target):
        if p == t:
            count += 1
    return round(count / length, 4)
