import os
import warnings
from typing import Union

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from loguru import logger
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.config import config
from src.distillation import DistillationLoss
from src.efficientnet_lite import build_efficientnet_lite
from src.ghost import ghost_net
from src.edgevit import EdgeViT_XXS, EdgeViT_XS, EdgeViT_S
from src.labelsmooth import LabelSmoothing
from src.mixup import Mixup
from src.no_bias_decay import add_weight_decay
from src.utils import accuracy_score, build_loader
from src.warmup import ExponentialWarmup
from timm.models import mobilenetv2_050, mobilenetv2_120d, mobilevitv2_100, mobilevitv2_150_384_in22ft1k, mobilevitv2_150, mobilevitv2_150_in22ft1k, efficientnet_lite3, ghostnet_130, fbnetv3_d
warnings.filterwarnings("ignore")

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

store_folder = f"cifar10_{config.model_name}{'_no_bias_decay' if config.no_bias_decay else ''}{'_warmup' if config.warmup else ''}{'_mixup' if config.mixup else ''}{'_label_smooth' if config.label_smooth else ''}{'_distillation' if config.distillation else ''}"
os.makedirs(store_folder, exist_ok=True)


trainloader, testloader = build_loader()

if "efficient" in config.model_name:
    """
    model = build_efficientnet_lite(
        config.model_name,
        num_classes=config.num_classes,
        group_normalization=config.group_normalization,
        weight_standardization=config.weight_standardization,
        stochastic_depth=config.stochastic_depth,
    )
    """
    model = efficientnet_lite3(True, num_classes=config.num_classes)
    logger.info(f"activate efficient_net")
    
elif "ghost" in config.model_name:
    model = ghostnet_130(True, num_classes=config.num_classes)
    logger.info(f"activate ghost_net")
elif "edge" in config.model_name or "EdgeViT" in config.model_name:
    model = EdgeViT_S(num_classes=config.num_classes)
    logger.info(f"activate edge_vit")
    
elif "mobilenetv2" in config.model_name:
    model = mobilenetv2_120d(True, num_classes=config.num_classes)
    logger.info(f"activate mobilenetv2")
elif "mobilevitv2" in config.model_name:
    model = mobilevitv2_150_in22ft1k(True, num_classes=config.num_classes)
    logger.info(f"activate mobilevitv2")
elif "fbnet" in config.model_name:
    model = fbnetv3_d(True, num_classes=config.num_classes)
    logger.info(f"activate fbnet") 
else:
    raise NotImplementedError

model.to(config.device)
model.train()


if config.no_bias_decay:
    params = add_weight_decay(model, 2e-5)
    optimizer = optim.AdamW(params, lr=config.lr, betas=(0.9, 0.999))
else:
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=2e-5
    )


num_steps = len(trainloader) * config.epochs * 2

lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
if config.warmup:
    warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=num_steps // 10)
    logger.info(r"Activated warmup")
if config.mixup:
    mixup_trainer = Mixup()
    logger.info(r"Activated mixup")

if config.distillation:
    distillation_trainer = DistillationLoss(4)
    logger.info(r"Activated distillation loss, please import teacher_network")

    """
    Example

    model_teacher = build_efficientnet_lite(
        'efficientnet_lite3',
        num_classes=10,
        group_normalization=config.group_normalization,
        weight_standardization=config.weight_standardization,
        stochastic_depth=config.stochastic_depth,)
    model_teacher.to(config.device)
    model_teacher.load(weight_path)
    model_teacher.eval()
    """
else:
    model_teacher = lambda N: None

criterion = LabelSmoothing(classes=config.num_classes, smoothing=config.label_smooth)


frequency = len(trainloader) // 3
writer = SummaryWriter(log_dir=store_folder)
trigger_mixup = config.epochs // 5
stop_mixup = (config.epochs // 5) * 4
for _ in range(config.epochs):
    loader = iter(trainloader)
    loss_ = []

    model.train()
    for iteration in range(len(trainloader)):
        optimizer.zero_grad()
        if config.warmup:
            lr_scheduler.step(lr_scheduler.last_epoch + 1)
            warmup_scheduler.step()
        else:
            lr_scheduler.step()

        mini_batch_images, mini_batch_labels = next(loader)
        mini_batch_images, mini_batch_labels = mini_batch_images.to(
            config.device
        ), mini_batch_labels.to(config.device)
        if config.mixup and trigger_mixup <= _ <= stop_mixup:
            loss = mixup_trainer(model, mini_batch_images, mini_batch_labels)
        else:
            output = model(mini_batch_images)
            loss = criterion(output, mini_batch_labels)

        if config.distillation:
            teacher_output = model_teacher(mini_batch_images)
            loss += distillation_trainer(output, teacher_output)

        loss.backward()
        optimizer.step()
        loss_.append(loss.detach().item())
        if iteration % frequency == 0:
            logger.info(
                f"epoch : {_}, iteration : {iteration}, loss : {round(np.mean(loss_), 4)}"
            )
    writer.add_scalar("train/loss", round(np.mean(loss_), 4), _)

    loader = iter(testloader)
    loss_ = []
    accuracy = []
    model.eval()
    for iteration in range(len(testloader)):
        mini_batch_images, mini_batch_labels = next(loader)
        mini_batch_images, mini_batch_labels = mini_batch_images.to(
            config.device
        ), mini_batch_labels.to(config.device)
        output = model(mini_batch_images)
        loss = criterion(output, mini_batch_labels)
        loss_.append(loss.detach().item())

        output = output.softmax(1).argmax(1).detach().cpu().numpy()
        mini_batch_labels = mini_batch_labels.detach().cpu().numpy()
        accuracy.append(accuracy_score(output, mini_batch_labels))

    writer.add_scalar("test/loss", round(np.mean(loss_), 4), _)
    writer.add_scalar("test/accuracy", round(np.mean(accuracy), 4), _)
    logger.info(
        f"epoch : {_}, test_loss : {round(np.mean(loss_), 4)}, test_accuracy : {round(np.mean(accuracy), 4)}"
    )
logger.info("Done")
