import numpy as np
import torch

from src.labelsmooth import LabelSmoothing


def test_smooth():
    criterion = LabelSmoothing(classes=6, smoothing=0.1)
    prediction = torch.tensor([0.4, 0.8, 20, 1, 2, 3], dtype=torch.float32).reshape(
        1, -1
    )
    ground_truth = torch.tensor([5], dtype=torch.long).reshape(-1, 1)
    criterion(prediction, ground_truth)
