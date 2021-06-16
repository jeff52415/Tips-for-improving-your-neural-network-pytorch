import torch
import torch.nn as nn
from torch.distributions.beta import Beta


class Mixup(nn.Module):
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()
        self.beta = Beta(alpha, alpha)

    def __get_mixup_batch(
        self, mini_batch_image: torch.Tensor, mini_batch_label: torch.Tensor
    ):
        lambda_ = self.beta.sample()
        batch_size = mini_batch_image.shape[0]
        shuffle = torch.randperm(batch_size)
        shuffle_mini_batch_x = mini_batch_image[shuffle]
        shuffle_mini_batch_y = mini_batch_label[shuffle]
        mixup_x = mini_batch_image * lambda_ + (1 - lambda_) * shuffle_mini_batch_x
        return (
            mixup_x,
            mini_batch_label.reshape(-1),
            shuffle_mini_batch_y.reshape(-1),
            lambda_,
        )

    def forward(
        self,
        model: nn.Module,
        mini_batch_image: torch.Tensor,
        mini_batch_label: torch.Tensor,
    ):
        """
        mini_batch_image -> shape == [batch, channel, height, width]
        mini_batch_label -> shape == [batch, 1] or [batch]
        """
        (
            mixup_x,
            mini_batch_label,
            shuffle_mini_batch_y,
            lambda_,
        ) = self.__get_mixup_batch(mini_batch_image, mini_batch_label)
        output = model(mixup_x)
        loss = self.criterion(output, mini_batch_label) * lambda_ + (
            1 - lambda_
        ) * self.criterion(output, shuffle_mini_batch_y)
        return loss
