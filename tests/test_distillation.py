import torch

from src.distillation import DistillationLoss


def test_distill():
    distillation_loss = DistillationLoss(4)
    teacher_output = torch.randn(1, 10)
    student_output = torch.randn(1, 10)
    distillation_loss(student_output, teacher_output)
