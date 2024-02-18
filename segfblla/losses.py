import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules.loss import _Loss


class GeneralizedDiceLoss(_Loss):
    """
    Generalized Dice Loss for multi-label classification.
    """
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()
        self.eps = 1e-7

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        assert y_true.size(0) == y_pred.size(0)

        y_pred = F.logsigmoid(y_pred).exp()

        num_classes = y_pred.size(1)
        y_true = y_true.permute(1, 0, 2, 3).view(num_classes, -1)
        y_pred = y_pred.permute(1, 0, 2, 3).view(num_classes, -1)

        # Find classes weights:
        sum_targets = y_true.sum(-1)
        class_weights = torch.reciprocal((sum_targets * sum_targets).clamp(min=self.eps))

        # Compute generalized Dice loss:
        intersection = ((y_pred * y_true).sum(-1) * class_weights).sum()
        cardinality = ((y_pred + y_true).sum(-1) * class_weights).sum()

        return 1 - 2 * (intersection / cardinality.clamp(min=self.eps))
