from typing import Optional, Callable, Union

import numpy as np
import torch
from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.utils import get_dictkey_auto_fn
from pytorch_toolbelt.utils import all_gather
from torch import Tensor


class MultilabelAccuracyMetricCallback(Callback):
    """
    Accuracy score metric for multi-label case (aka Exact Match Ratio, Subset accuracy).
    """

    def __init__(
        self,
        outputs_to_labels: Union[float, Callable[[Tensor], Tensor]],
        targets_key: str = "targets",
        predictions_key: str = "logits",
        prefix: str = "metrics/subset_accuracy",
        log_per_batch: bool = False,
        label_dim: int = 1,
    ):
        """
        Args:
            targets_key: input key to use for accuracy calculation;
                specifies our `y_true`
            predictions_key: output key to use for accuracy calculation;
                specifies our `y_pred`
            prefix: key for the metric's name
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.predictions_key = predictions_key
        self.targets_key = targets_key
        self.outputs_to_labels = outputs_to_labels
        self.label_dim = label_dim
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
        predictions = runner.output[self.predictions_key]
        targets = runner.input[self.targets_key]

        if isinstance(self.outputs_to_labels, float):
            predictions = predictions > self.outputs_to_labels
        elif callable(self.outputs_to_labels):
            predictions = self.outputs_to_labels(predictions)

        if predictions.size() != targets.size():
            raise RuntimeError(
                "Shape of predictions and targets must be equal. Got {predictions.size()} and {targets.size()}."
            )

        correct_preds: Tensor = predictions == targets
        batch_correct: Tensor = correct_preds.all(dim=self.label_dim, keepdim=False)

        batch_correct = float(batch_correct.sum())
        batch_totals = int(batch_correct.numel())

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
