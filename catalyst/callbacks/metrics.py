from functools import partial
from typing import Optional

import numpy as np
import torch
from pytorch_toolbelt.utils import all_gather, to_numpy

from catalyst.core import Callback, CallbackOrder, IRunner

__all__ = [
    "BINARY_MODE",
    "IoUMetricsCallback",
    "MULTICLASS_MODE",
    "MULTILABEL_MODE",
    "binary_dice_iou_score",
    "multiclass_dice_iou_score",
    "multilabel_dice_iou_score",
]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def binary_dice_iou_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold: Optional[float] = None,
    nan_score_on_empty=False,
    eps: float = 1e-7,
    ignore_index=None,
) -> float:
    """
    Compute IoU score between two image tensors
    :param y_pred: Input image tensor of any shape
    :param y_true: Target image of any shape (must match size of y_pred)
    :param mode: Metric to compute (dice, iou)
    :param threshold: Optional binarization threshold to apply on @y_pred
    :param nan_score_on_empty: If true, return np.nan if target has no positive pixels;
        If false, return 1. if both target and input are empty, and 0 otherwise.
    :param eps: Small value to add to denominator for numerical stability
    :param ignore_index:
    :return: Float scalar
    """
    assert mode in {"dice", "iou"}

    # Make binary predictions
    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)

    if ignore_index is not None:
        mask = (y_true != ignore_index).to(y_true.dtype)
        y_true = y_true * mask
        y_pred = y_pred * mask

    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    if mode == "dice":
        score = (2.0 * intersection) / (cardinality + eps)
    else:
        score = intersection / (cardinality - intersection + eps)

    has_targets = torch.sum(y_true) > 0
    has_predicted = torch.sum(y_pred) > 0

    if not has_targets:
        if nan_score_on_empty:
            score = np.nan
        else:
            score = float(not has_predicted)
    return score


def multiclass_dice_iou_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold=None,
    eps=1e-7,
    nan_score_on_empty=False,
    classes_of_interest=None,
    ignore_index=None,
):
    ious = []
    num_classes = y_pred.size(0)
    y_pred = y_pred.argmax(dim=0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        y_pred_i = (y_pred == class_index).float()
        y_true_i = (y_true == class_index).float()
        if ignore_index is not None:
            not_ignore_mask = (y_true != ignore_index).float()
            y_pred_i *= not_ignore_mask
            y_true_i *= not_ignore_mask

        iou = binary_dice_iou_score(
            y_pred=y_pred_i,
            y_true=y_true_i,
            mode=mode,
            nan_score_on_empty=nan_score_on_empty,
            threshold=threshold,
            eps=eps,
        )
        ious.append(iou)

    return ious


def multilabel_dice_iou_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mode="dice",
    threshold=None,
    eps=1e-7,
    nan_score_on_empty=False,
    classes_of_interest=None,
    ignore_index=None,
):
    ious = []
    num_classes = y_pred.size(0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        iou = binary_dice_iou_score(
            y_pred=y_pred[class_index],
            y_true=y_true[class_index],
            mode=mode,
            threshold=threshold,
            nan_score_on_empty=nan_score_on_empty,
            eps=eps,
            ignore_index=ignore_index,
        )
        ious.append(iou)

    return ious


class IoUMetricsCallback(Callback):
    """
    A metric callback for computing either Dice or Jaccard metric
    which is computed across whole epoch, not per-batch.
    """

    def __init__(
        self,
        mode: str,
        metric="dice",
        class_names=None,
        classes_of_interest=None,
        input_key: str = "targets",
        output_key: str = "logits",
        nan_score_on_empty=True,
        prefix: str = None,
        ignore_index=None,
    ):
        """
        :param mode: One of: 'binary', 'multiclass', 'multilabel'.
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.Metric)
        if mode not in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}:
            raise ValueError("Mode must be one of BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE")

        if prefix is None:
            prefix = metric

        if classes_of_interest is not None:
            if classes_of_interest.dtype == np.bool:
                num_classes = len(classes_of_interest)
                classes_of_interest = np.arange(num_classes)[classes_of_interest]

            if class_names is not None:
                if len(class_names) != len(classes_of_interest):
                    raise ValueError(
                        "Length of 'classes_of_interest' must be equal to length of 'classes_of_interest'"
                    )

        self.mode = mode
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.class_names = class_names
        self.classes_of_interest = classes_of_interest
        self.scores = []

        if self.mode == BINARY_MODE:
            self.score_fn = partial(
                binary_dice_iou_score,
                threshold=0.0,
                nan_score_on_empty=nan_score_on_empty,
                mode=metric,
                ignore_index=ignore_index,
            )

        if self.mode == MULTICLASS_MODE:
            self.score_fn = partial(
                multiclass_dice_iou_score,
                mode=metric,
                threshold=0.0,
                nan_score_on_empty=nan_score_on_empty,
                classes_of_interest=self.classes_of_interest,
                ignore_index=ignore_index,
            )

        if self.mode == MULTILABEL_MODE:
            self.score_fn = partial(
                multilabel_dice_iou_score,
                mode=metric,
                threshold=0.0,
                nan_score_on_empty=nan_score_on_empty,
                classes_of_interest=self.classes_of_interest,
                ignore_index=ignore_index,
            )

    def on_loader_start(self, state):
        self.scores = []

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        outputs = runner.output[self.output_key].detach()
        targets = runner.input[self.input_key].detach()

        batch_size = targets.size(0)
        score_per_image = []
        for image_index in range(batch_size):
            score_per_class = self.score_fn(y_pred=outputs[image_index], y_true=targets[image_index])
            score_per_class = to_numpy(score_per_class).reshape(-1)
            score_per_image.append(score_per_class)

        mean_score = np.nanmean(score_per_image)
        runner.batch_metrics[self.prefix] = float(mean_score)
        self.scores.extend(score_per_image)

    def on_loader_end(self, runner: IRunner):
        scores = np.concatenate(all_gather(np.array(self.scores)))
        mean_per_class = np.nanmean(scores, axis=1)  # Average across classes
        mean_score = np.nanmean(mean_per_class, axis=0)  # Average across images

        runner.loader_metrics[self.prefix] = float(mean_score)

        # Log additional IoU scores per class
        if self.mode in {MULTICLASS_MODE, MULTILABEL_MODE}:
            num_classes = scores.shape[1]
            class_names = self.class_names
            if class_names is None:
                class_names = [f"class_{i}" for i in range(num_classes)]

            scores_per_class = np.nanmean(scores, axis=0)
            for class_name, score_per_class in zip(class_names, scores_per_class):
                runner.loader_metrics[self.prefix + "_" + class_name] = float(score_per_class)
