import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, student_temp: float = 0.1,
                 teacher_temp: float = 0.04, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output: torch.Tensor,
                teacher_output: torch.Tensor) -> torch.Tensor:
        teacher_centered = teacher_output - self.center
        teacher_logits = teacher_centered / self.teacher_temp
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        teacher_probs = teacher_probs.detach()

        student_logits = student_output / self.student_temp
        student_probs = F.log_softmax(student_logits, dim=-1)

        loss = -(teacher_probs * student_probs).sum(dim=-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center_momentum * self.center + (1 - self.center_momentum) * batch_center
