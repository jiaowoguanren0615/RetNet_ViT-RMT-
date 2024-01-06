"""
Implements the knowledge distillation loss, proposed in deit
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class SoftTargetCrossEntropy(nn.Module):
    """
    The native CE loss with soft target
    input: x is output of model, target is ground truth
    return: loss
    """
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N == N_rep:
            target = target.repeat(N_rep // N, 1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class TokenLabelGTCrossEntropy(nn.Module):
    """
    Token labeling dense loss with ground gruth, see more from token labeling
    input: x is output of model, target is ground truth
    return: loss
    """
    def __init__(self,
                 dense_weight=1.0,
                 cls_weight=1.0,
                 mixup_active=True,
                 smoothing=0.1,
                 classes=1000):
        super(TokenLabelGTCrossEntropy, self).__init__()

        self.CE = SoftTargetCrossEntropy()

        self.dense_weight = dense_weight
        self.smoothing = smoothing
        self.mixup_active = mixup_active
        self.classes = classes
        self.cls_weight = cls_weight
        assert dense_weight + cls_weight > 0

    def forward(self, x, target):

        output, aux_output, bb = x
        bbx1, bby1, bbx2, bby2 = bb

        B, N, C = aux_output.shape
        if len(target.shape) == 2:
            target_cls = target
            target_aux = target.repeat(1, N).reshape(B * N, C)
        else:
            ground_truth = target[:, :, 0]
            target_cls = target[:, :, 1]
            ratio = (0.9 - 0.4 *
                     (ground_truth.max(-1)[1] == target_cls.max(-1)[1])
                     ).unsqueeze(-1)
            target_cls = target_cls * ratio + ground_truth * (1 - ratio)
            target_aux = target[:, :, 2:]
            target_aux = target_aux.transpose(1, 2).reshape(-1, C)
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / N)
        if lam < 1:
            target_cls = lam * target_cls + (1 - lam) * target_cls.flip(0)

        aux_output = aux_output.reshape(-1, C)

        loss_cls = self.CE(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)

        return self.cls_weight * loss_cls + self.dense_weight * loss_aux


class TokenLabelSoftTargetCrossEntropy(nn.Module):
    """
    Token labeling dense loss with soft target, see more from token labeling
    input: x is output of model, target is ground truth
    return: loss
    """
    def __init__(self):
        super(TokenLabelSoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N == N_rep:
            target = target.repeat(N_rep // N, 1)
        if len(target.shape) == 3 and target.shape[-1] == 2:
            target = target[:, :, 1]
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class TokenLabelCrossEntropy(nn.Module):
    """
    Token labeling loss without ground truth
    input: x is output of model, target is ground truth
    return: loss
    """
    def __init__(self,
                 dense_weight=1.0,
                 cls_weight=1.0,
                 mixup_active=True,
                 classes=1000):
        """
        Constructor Token labeling loss.
        """
        super(TokenLabelCrossEntropy, self).__init__()

        self.CE = SoftTargetCrossEntropy()

        self.dense_weight = dense_weight
        self.mixup_active = mixup_active
        self.classes = classes
        self.cls_weight = cls_weight
        assert dense_weight + cls_weight > 0

    def forward(self, x, target):

        output, aux_output, bb = x
        bbx1, bby1, bbx2, bby2 = bb

        B, N, C = aux_output.shape
        if len(target.shape) == 2:
            target_cls = target
            target_aux = target.repeat(1, N).reshape(B * N, C)
        else:
            target_cls = target[:, :, 1]
            target_aux = target[:, :, 2:]
            target_aux = target_aux.transpose(1, 2).reshape(-1, C)
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / N)
        if lam < 1:
            target_cls = lam * target_cls + (1 - lam) * target_cls.flip(0)

        aux_output = aux_output.reshape(-1, C)
        loss_cls = self.CE(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)
        return self.cls_weight * loss_cls + self.dense_weight * loss_aux