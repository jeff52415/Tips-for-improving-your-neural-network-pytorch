from typing import Union

import torch
import torch.nn as nn


class DistillationLoss(nn.Module):
    def __init__(self, temperature: Union[int, float] = 2):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.temperature = temperature
        self.softmax = nn.Softmax(1)
        self.logsoftmax = nn.LogSoftmax(1)

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor):
        """
        student_output  -> shape == [batch, class_number]
        teacher_output -> shape == [batch, class_number]

        """

        if student_output.shape != teacher_output.shape:
            raise ValueError(
                f"Mismatch between student_output and teacher_output, got student_output : {student_output.shape} & teacher_output : {teacher_output.shape} respectively"
            )
        student_output = self.logsoftmax(student_output / self.temperature)
        teacher_output = self.softmax(teacher_output / self.temperature)

        loss = (
            self.criterion(student_output, teacher_output)
            * self.temperature
            * self.temperature
        )
        return loss
