from typing import Callable, Optional

import numpy as np
import torch
import typing
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn

from catalyst.core import Callback, CallbackOrder
from catalyst.utils import get_dictkey_auto_fn

__all__ = ["RocAucMetricCallback"]

from pytorch_toolbelt.utils import to_numpy
from pytorch_toolbelt.utils.distributed import all_gather, is_main_process
from pytorch_toolbelt.modules import instantiate_activation_block
from catalyst.utils import get_tensorboard_logger


class RocAucMetricCallback(Callback):
    """
    Roc Auc score metric
    """

    def __init__(
        self,
        outputs_to_probas: Optional[Callable[[Tensor], Tensor]] = torch.sigmoid,
        targets_key: str = "targets",
        predictions_key: str = "logits",
        metric_name: str = "metrics/roc_auc",
        average: str = "macro",
        ignore_index: Optional[int] = None,
        log_pr_curve: bool = True,
        fix_nans: bool = False,
        score_only_present_classes: bool = False,
    ):
        """
        Args:
            targets_key: input key to use for accuracy calculation;
                specifies our `y_true`
            predictions_key: output key to use for accuracy calculation;
                specifies our `y_pred`
            metric_name: key for the metric's name
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
        self.metric_name = metric_name
        self.predictions_key = predictions_key
        self.targets_key = targets_key
        self.ignore_index = ignore_index
        self.outputs_to_probas = outputs_to_probas
        self.y_trues = []
        self.y_preds = []
        self.average = average
        self.log_pr_curve = log_pr_curve
        self.fix_nans = fix_nans
        self._get_targets = get_dictkey_auto_fn(targets_key)
        self._get_predictions = get_dictkey_auto_fn(predictions_key)
        self.score_only_present_classes = score_only_present_classes

    def on_loader_start(self, state):
        self.y_trues = []
        self.y_preds = []

    @torch.no_grad()
    def on_batch_end(self, runner):
        pred_probas = runner.output[self.predictions_key].float()
        true_labels = runner.input[self.targets_key].float()

        if self.outputs_to_probas is not None:
            pred_probas = self.outputs_to_probas(pred_probas)

        y_trues = to_numpy(true_labels)
        y_preds = to_numpy(pred_probas)

        if self.ignore_index is not None:
            y_trues = y_trues.reshape(-1)
            y_preds = y_preds.reshape(-1)
            include_mask = y_trues != self.ignore_index
            y_trues = y_trues[include_mask]
            y_preds = y_preds[include_mask]

        # Aggregate flattened labels
        self.y_trues.extend(y_trues)
        self.y_preds.extend(y_preds)

    def on_loader_end(self, runner):
        y_trues = np.concatenate(all_gather(self.y_trues))
        y_preds = np.concatenate(all_gather(self.y_preds))

        if self.fix_nans:
            y_preds[~np.isfinite(y_preds)] = 0.5

        if self.score_only_present_classes:
            mask = y_trues.sum(axis=0) > 0
            y_trues = y_trues[:, mask]
            y_preds = y_preds[:, mask]

        score = roc_auc_score(y_true=y_trues, y_score=y_preds, average=self.average)
        runner.loader_metrics[self.metric_name] = float(score)

        if self.log_pr_curve and is_main_process():
            logger = get_tensorboard_logger(runner)
            logger.add_pr_curve(
                self.metric_name,
                predictions=y_preds,
                labels=y_trues,
                global_step=runner.global_epoch,
                num_thresholds=255,
            )
