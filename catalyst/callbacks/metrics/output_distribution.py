import typing
from typing import Optional, Callable

import numpy as np
import torch
from torch import Tensor, nn
from pytorch_toolbelt.utils import all_gather, to_numpy, is_main_process
from pytorch_toolbelt.modules import instantiate_activation_block
from catalyst.core import IRunner, Callback, CallbackOrder
from catalyst.utils import get_tensorboard_logger


class OutputDistributionCallback(Callback):
    """
    Plot histogram of predictions for each class. This callback supports binary & multi-classs predictions
    """

    def __init__(
        self,
        targets_key: str,
        predictions_key: str,
        outputs_to_probas: Optional[Callable[[Tensor], Tensor]],
        num_classes: int,
        prefix="distribution",
        ignore_index=None,
    ):
        """

        Args:
            targets_key:
            predictions_key:
            output_activation: A function that should convert logits to class labels
            For binary predictions this could be `lambda x: int(x > 0.5)` or `lambda x: torch.argmax(x, dim=1)`
            for multi-class predictions.
            num_classes: Number of classes. Must be 2 for binary.
            prefix:
        """
        if outputs_to_probas is None:
            outputs_to_probas = nn.Identity()
        elif isinstance(outputs_to_probas, str):
            outputs_to_probas = instantiate_activation_block(outputs_to_probas)
        elif isinstance(outputs_to_probas, typing.Callable):
            outputs_to_probas = outputs_to_probas
        else:
            raise ValueError(f"Unsupported type of outputs_to_probas={outputs_to_probas}")

        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.targets_key = targets_key
        self.output_key = predictions_key
        self.true_labels = []
        self.pred_labels = []
        self.num_classes = num_classes
        self.outputs_to_probas = outputs_to_probas
        self.ignore_index = ignore_index

    def on_loader_start(self, state: IRunner):
        self.true_labels = []
        self.pred_labels = []

    @torch.no_grad()
    def on_batch_end(self, state: IRunner):
        y_trues = state.input[self.targets_key].detach()
        y_preds = state.output[self.output_key].detach().float()
        if self.outputs_to_probas is not None:
            y_preds = self.outputs_to_probas(y_preds)

        y_trues = to_numpy(y_trues).reshape(-1)
        y_preds = to_numpy(y_preds).reshape(-1)

        if self.ignore_index is not None:
            include_mask = y_trues != self.ignore_index
            y_trues = y_trues[include_mask]
            y_preds = y_preds[include_mask]

        self.true_labels.extend(y_trues)
        self.pred_labels.extend(y_preds)

    def on_loader_end(self, state: IRunner):
        true_labels = np.concatenate(all_gather(np.array(self.true_labels)))
        pred_probas = np.concatenate(all_gather(np.array(self.pred_labels)))

        if is_main_process():
            logger = get_tensorboard_logger(state)

            for class_label in range(self.num_classes):
                p = pred_probas[true_labels == class_label]
                if p.any():
                    logger.add_histogram(
                        tag=f"{self.prefix}/{class_label}",
                        values=pred_probas[true_labels == class_label],
                        global_step=state.global_epoch,
                    )
