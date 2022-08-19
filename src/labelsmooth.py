from typing import Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
#from multipledispatch import dispatch


class LabelSmoothing(nn.Module):
    def __init__(self, classes: int, smoothing: float = 0.1):
        super().__init__()
        assert 0 <= smoothing < 1
        if not smoothing:
            self.criterion_entropy = nn.CrossEntropyLoss()
            logger.info("Deactivated smoothing, apply CrossEntropyLoss.")
        else:
            self.criterion = nn.KLDivLoss(reduction="batchmean")
            logger.info("Activated smoothing, apply KLDivLoss.")

        self.classes = classes
        self.smoothing = smoothing
        self.confidence = 1 - smoothing

    @torch.no_grad()
    def convert_label_to_smooth(self, true_labels: torch.Tensor):
        label_shape = torch.Size((true_labels.size(0), self.classes))
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(self.smoothing / (self.classes - 1))
        true_dist = true_dist.scatter_(1, true_labels.data, self.confidence)
        return true_dist

    # @dispatch(torch.Tensor, torch.Tensor)
    def forward(
        self, prediction: torch.Tensor, true_labels: Union[np.ndarray, torch.Tensor]
    ):
        """
        prediction  -> shape == [batch, self.classes]
        true_labels -> shape == [batch, 1]

        """
        if prediction.shape[1] != self.classes:
            raise ValueError(
                f"Mismatch between prediction and specified class number, have pre-specified classes equal to {self.classes} while {prediction.shape[1]} in prediction"
            )
        if not isinstance(true_labels, torch.Tensor):
            true_labels = torch.tensor(true_labels, dtype=torch.long)
        else:
            true_labels = true_labels.type(torch.long)
        if self.smoothing:
            true_labels = self.convert_label_to_smooth(true_labels.reshape(-1, 1))
            prediction = prediction.log_softmax(-1)
            loss = self.criterion(prediction, true_labels)
        else:
            loss = self.criterion_entropy(prediction, true_labels.reshape(-1))
        return loss
