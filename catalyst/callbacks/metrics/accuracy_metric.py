from typing import Callable, Optional

import numpy as np
import torch
from catalyst.core import IRunner, Callback, CallbackOrder
from catalyst.utils import get_dictkey_auto_fn
from pytorch_toolbelt.utils import all_gather
from torch import Tensor


class AccuracyMetricCallback(Callback):
    """
    Accuracy metric callback.
    """

    def __init__(
        self,
        outputs_to_labels: Callable[[Tensor], Tensor],
        targets_key: str = "targets",
        predictions_key: str = "logits",
        prefix: str = "metrics/accuracy",
        log_per_batch: bool = False,
        ignore_index: Optional[int] = None,
    ):
        """
        Args:
            targets_key: input key to use for accuracy calculation;
                specifies our `y_true`
            predictions_key: output key to use for accuracy calculation;
                specifies our `y_pred`
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.predictions_key = predictions_key
        self.targets_key = targets_key
        self.ignore_index = ignore_index
        self.outputs_to_labels = outputs_to_labels
        self.correct = 0
        self.totals = 0
        self.log_per_batch = log_per_batch
        self._get_targets = get_dictkey_auto_fn(targets_key)
        self._get_predictions = get_dictkey_auto_fn(predictions_key)

    def on_loader_start(self, state):
        self.correct = 0
        self.totals = 0

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        predictions = self.outputs_to_labels(runner.output[self.predictions_key])
        true_labels = runner.input[self.targets_key].type_as(pred_labels)

        if isinstance(self.outputs_to_labels, float):
            predictions = predictions > self.outputs_to_labels
        elif callable(self.outputs_to_labels):
            predictions = self.outputs_to_labels(predictions)

        true_labels = true_labels.view(-1)
        pred_labels = pred_labels.view(-1)

        correct_mask = pred_labels == true_labels

        if self.ignore_index is not None:
            mask = true_labels != self.ignore_index
            correct_mask = correct_mask * mask
            batch_totals = int(mask.sum())
        else:
            batch_totals = len(true_labels)

        batch_correct = int(correct_mask.sum())

        self.correct += batch_correct
        self.totals += batch_totals

        if self.log_per_batch:
            batch_accuracy = float(batch_correct) / float(batch_totals)
            runner.batch_metrics[self.prefix + "/batch"] = batch_accuracy

    def on_loader_end(self, runner: IRunner):
        correct = np.sum(all_gather(self.correct))
        total = np.sum(all_gather(self.totals))
        accuracy = float(correct) / float(total)
        runner.loader_metrics[self.prefix] = accuracy
