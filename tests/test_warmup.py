from torch import optim
from torchvision import models

from src.warmup import ExponentialWarmup


def test_warmup():
    model = models.resnet18()
    num_steps = 20
    optimizer = optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps
    )  # T_max step 之後會重新
    warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=num_steps // 10)
    result = []
    for epoch in range(num_steps):
        # https://github.com/pytorch/pytorch/blob/cf38b20c61e2b08496c51b0d879892f388d6e03b/torch/optim/lr_scheduler.py#L456-L458
        # give epoch to refer base_lrs
        lr_scheduler.step(lr_scheduler.last_epoch + 1)
        warmup_scheduler.step()
        result.append(optimizer.param_groups[0]["lr"])
