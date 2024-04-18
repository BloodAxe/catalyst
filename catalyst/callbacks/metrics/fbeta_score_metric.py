from typing import Optional, Callable

import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix
from torch import Tensor

from catalyst.core import Callback, CallbackOrder, IRunner
from pytorch_toolbelt.utils import to_numpy, all_gather

from catalyst.core.callback import CallbackUtils


class FBetaScoreCallback(Callback):
    """
    Compute FBeta metric score

    """

    def __init__(
        self,
        outputs_transform_fn: Optional[Callable[[Tensor], Tensor]],
        targets_transforms_fn: Optional[Callable[[Tensor], Tensor]],
        num_classes: int,
        targets_key: str,
        predictions_key: str,
        metric_name: str,
        beta: float = 1,
        average="macro",
        ignore_index: Optional[int] = None,
        zero_division="warn",
    ):
        """
        :param targets_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param predictions_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.Metric)
        self.num_classes = num_classes
        self.prefix = metric_name
        self.predictions_key = predictions_key
        self.targets_key = targets_key
        self.ignore_index = ignore_index
        self.average = average
        self.confusion_matrix = None
        self.beta = beta
        self.zero_division = zero_division
        self.outputs_transform_fn = CallbackUtils.get_transform_fn(outputs_transform_fn)
        self.targets_transforms_fn = CallbackUtils.get_transform_fn(targets_transforms_fn)

    def on_loader_start(self, state):
        self.confusion_matrix = np.zeros((self.num_classes, 2, 2), dtype=np.long)

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        predictions = self.outputs_transform_fn(runner.output[self.predictions_key])
        targets = self.targets_transforms_fn(runner.input[self.targets_key])

        if predictions.size() != targets.size():
            raise RuntimeError(
                "Shape of predictions and targets must be equal. Got {predictions.size()} and {targets.size()}."
            )

        targets = targets.view(-1)
        predictions = predictions.view(-1)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            predictions = torch.masked_select(predictions, mask)
            targets = torch.masked_select(targets, mask)

        if len(targets):
            targets = to_numpy(targets)
            predictions = to_numpy(predictions)
            batch_cm = multilabel_confusion_matrix(
                y_true=targets, y_pred=predictions, labels=np.arange(self.num_classes, dtype=int)
            )
            self.confusion_matrix = self.confusion_matrix + batch_cm

    def on_loader_end(self, runner: IRunner):
        MCM = np.sum(all_gather(self.confusion_matrix), axis=0)
        metric = self._f1_from_confusion_matrix(
            MCM, beta=self.beta, average=self.average, zero_division=self.zero_division
        )
        runner.loader_metrics[self.prefix] = float(metric)

    def _f1_from_confusion_matrix(
        self,
        mcm: np.ndarray,
        average: str,
        beta: float,
        warn_for=("precision", "recall", "f-score"),
        zero_division="warn",
    ):
        """
        Code borrowed from sklear.metrics
        """
        tp_sum = mcm[:, 1, 1]
        pred_sum = tp_sum + mcm[:, 0, 1]
        true_sum = tp_sum + mcm[:, 1, 0]

        if average == "micro":
            tp_sum = np.array([tp_sum.sum()])
            pred_sum = np.array([pred_sum.sum()])
            true_sum = np.array([true_sum.sum()])

        # Finally, we have all our sufficient statistics. Divide! #
        beta2 = beta**2

        # Divide, and on zero-division, set scores and/or warn according to
        # zero_division:
        from sklearn.metrics._classification import _prf_divide, _warn_prf

        precision = _prf_divide(tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division)
        recall = _prf_divide(tp_sum, true_sum, "recall", "true", average, warn_for, zero_division)

        # warn for f-score only if zero_division is warn, it is in warn_for
        # and BOTH prec and rec are ill-defined
        if zero_division == "warn" and ("f-score",) == warn_for:
            if (pred_sum[true_sum == 0] == 0).any():
                _warn_prf(average, "true nor predicted", "F-score is", len(true_sum))

        # if tp == 0 F will be 1 only if all predictions are zero, all labels are
        # zero, and zero_division=1. In all other case, 0
        if np.isposinf(beta):
            f_score = recall
        else:
            denom = beta2 * precision + recall

            denom[denom == 0.0] = 1  # avoid division by 0
            f_score = (1 + beta2) * precision * recall / denom

        # Average the results
        if average == "weighted":
            weights = true_sum
            if weights.sum() == 0:
                zero_division_value = 0.0 if zero_division in ["warn", 0] else 1.0
                # precision is zero_division if there are no positive predictions
                # recall is zero_division if there are no positive labels
                # fscore is zero_division if all labels AND predictions are
                # negative
                return (
                    zero_division_value if pred_sum.sum() == 0 else 0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0,
                    None,
                )
        else:
            weights = None

        if average is not None:
            assert average != "binary" or len(precision) == 1
            f_score = np.average(f_score, weights=weights)

        return f_score
