# since there is no built-in hinge loss in PyTorch, I thought it would be beneficial for me
# if I define it by myself.

import torch
import torch.nn as nn
from torch import Tensor


class HingeLoss(nn.Module):
    def __init__(self) -> None:
        super(HingeLoss, self).__init__()

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Hinge Loss Function
        commonly used in Support Vector Machines
        params:
            -y_true: (Tensor, optional), true labels (-1 or 1 for binary classification.).
            -y_pred: (Tensor, optional), predicted scores or outputs of the model.


        returns:
            -loss: Tensor, calculated hinge loss
        """
        return torch.mean(torch.max(torch.zeros_like(y_true), 1 - y_true * y_pred))

# it does look like BCELoss (built-in torch) since it also works at binary classification
# however, it is not. Hinge Loss encourages maximizing margins between classes
# by penalizing the model with higher loss values when data points are missclassified
# or when the margin between the correct class and the other classes small, and suitable for SVMs
# BCELoss works with predicted probabilities, while Hinge Loss works with raw outputs (raw outputs == scores == logits).
