# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from functools import partial

import torch
from torch import nn

from catalyst import metrics


class DiceLoss(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()

        self.loss_fn = partial(
            metrics.dice, eps=eps, threshold=threshold, activation=activation
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Calculates loss between ``logits`` and ``target`` tensors.

        Args:
            logits: model logits
            targets: ground truth labels

        Returns:
            computed loss
        """
        dice = self.loss_fn(logits, targets)
        return 1 - dice




__all__ = ["DiceLoss"]
