from typing import Callable, List, Optional

import numpy as np
import torch
from catalyst.callbacks.visualization import get_tensorboard_logger
from catalyst.core import Callback, CallbackOrder, IRunner
from pytorch_toolbelt.utils import is_main_process, all_gather, plot_confusion_matrix, render_figure_to_tensor
from torch import Tensor


class ConfusionMatrixCallback(Callback):
    """
    Compute and log confusion matrix to Tensorboard.
    For use with Multiclass classification/segmentation.
    """

    def __init__(
        self,
        outputs_to_labels: Callable[[Tensor], Tensor],
        targets_key: str = "targets",
        predictions_key: str = "logits",
        prefix: str = "confusion_matrix",
        class_names: List[str] = None,
        num_classes: int = None,
        ignore_index: Optional[int] = None,
    ):
        """
        :param targets_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param predictions_key: output key to use for precision calculation;
            specifies our `y_pred`.
        :param ignore_index: same meaning as in nn.CrossEntropyLoss
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.class_names = class_names
        self.num_classes = num_classes if class_names is None else len(class_names)
        self.output_key = predictions_key
        self.targets_key = targets_key
        self.ignore_index = ignore_index
        self.confusion_matrix = None
        self.outputs_to_labels = outputs_to_labels

    def on_loader_start(self, state):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.long)

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

        true_labels = true_labels.view(-1)
        pred_labels = pred_labels.view(-1)

        if self.ignore_index is not None:
            mask = true_labels != self.ignore_index
            pred_labels = torch.masked_select(pred_labels, mask)
            true_labels = torch.masked_select(true_labels, mask)

        if len(true_labels):
            true_labels = to_numpy(true_labels)
            pred_labels = to_numpy(pred_labels)
            batch_cm = confusion_matrix(
                y_true=true_labels, y_pred=pred_labels, labels=np.arange(self.num_classes, dtype=int)
            )
            self.confusion_matrix = self.confusion_matrix + batch_cm

    def on_loader_end(self, runner: IRunner):
        if self.class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]
        else:
            class_names = self.class_names

        num_classes = len(class_names)
        cm = np.sum(all_gather(self.confusion_matrix), axis=0)

        if is_main_process():
            fig = plot_confusion_matrix(
                cm,
                figsize=(6 + num_classes // 3, 6 + num_classes // 3),
                class_names=class_names,
                normalize=True,
                noshow=True,
            )
            fig = render_figure_to_tensor(fig)

            logger = get_tensorboard_logger(runner)
            logger.add_image(f"{self.prefix}/epoch", fig, global_step=runner.global_epoch)
