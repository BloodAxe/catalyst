from typing import Optional, Callable

import numpy as np
import torch
from torch import Tensor

from catalyst.core import Callback, CallbackOrder, IRunner


class F1ScoreCallback(Callback):
    """
    Compute F1 metric score

    """

    def __init__(
        self,
        num_classes: int,
        outputs_to_labels: Callable[[Tensor], Tensor],
        targets_key: str = "targets",
        predictions_key: str = "logits",
        prefix: str = "f1",
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
        self.prefix = prefix
        self.predictions_key = predictions_key
        self.targets_key = targets_key
        self.ignore_index = ignore_index
        self.outputs_to_labels = outputs_to_labels
        self.average = average
        self.confusion_matrix = None
        self.zero_division = zero_division

    def on_loader_start(self, state):
        self.confusion_matrix = np.zeros((self.num_classes, 2, 2), dtype=np.long)

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        pred_labels = self.outputs_to_labels(runner.output[self.predictions_key])
        true_labels = runner.input[self.targets_key].type_as(pred_labels)

        true_labels = true_labels.view(-1)
        pred_labels = pred_labels.view(-1)

        if self.ignore_index is not None:
            mask = true_labels != self.ignore_index
            pred_labels = torch.masked_select(pred_labels, mask)
            true_labels = torch.masked_select(true_labels, mask)

        if len(true_labels):
            true_labels = to_numpy(true_labels)
            pred_labels = to_numpy(pred_labels)
            batch_cm = multilabel_confusion_matrix(
                y_true=true_labels, y_pred=pred_labels, labels=np.arange(self.num_classes, dtype=int)
            )
            self.confusion_matrix = self.confusion_matrix + batch_cm

    def on_loader_end(self, runner: IRunner):
        MCM = np.sum(all_gather(self.confusion_matrix), axis=0)
        metric = self._f1_from_confusion_matrix(MCM, average=self.average, zero_division=self.zero_division)
        runner.loader_metrics[self.prefix] = metric

    def _f1_from_confusion_matrix(
        self, MCM, average, beta=1, warn_for=("precision", "recall", "f-score"), zero_division="warn"
    ):
        """
        Code borrowed from sklear.metrics
        """
        tp_sum = MCM[:, 1, 1]
        pred_sum = tp_sum + MCM[:, 0, 1]
        true_sum = tp_sum + MCM[:, 1, 0]

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
