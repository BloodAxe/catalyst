import functools
from typing import Union, List, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_toolbelt.modules import instantiate_activation_block
from pytorch_toolbelt.utils import all_gather, to_numpy, is_main_process
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from catalyst.callbacks.metrics.segmentation_utils import SegmentationMeter
from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.utils import get_tensorboard_logger


class GlobalBinaryDiceScore(Callback):
    """
    Metric callback to compute global binary dice score.
    Global binary dice score is computed for all the pixels in the entire loader.
    """

    def __init__(
        self,
        predictions_key: Union[str, int, None],
        targets_key: Union[str, int, None],
        activation: Union[None, str, Callable[[Tensor], Tensor], nn.Module] = torch.sigmoid,
        threshold: Union[float, List[float], np.ndarray] = 0.5,
        metric_name: str = "metrics/global_dice",
        metric_threshold_name="metrics/global_dice_threshold",
        beta: float = 1.0,
        ignore_index: Optional[int] = None,
        targets_threshold: float = 0.5,
    ):
        """
        :param predictions_key: name of the key in ``runner.output`` dictionary with predictions
        :param targets_key: name of the key in ``runner.input`` dictionary with targets
        :param activation: An torch activation torch.nn.functional or torch module
        :param threshold: threshold for outputs binarization.
            Can be a single scalar or a list of thresholds to try.
            If a list is provided, callback logs the best value and corresponding threshold.
            Additionally, a PR-curve plot and curve (thresholds vs metric) plot is created.
        :param metric_name: name of the metric to display in the logs
        :param beta: beta parameter for F-measure computation
        :param ignore_index: If not None, targets with given index are ignored during metric computation
        :param targets_threshold: A threshold for targets binarization. Default: 0.5


        """
        super().__init__(CallbackOrder.Metric)
        self.metric_name = metric_name
        self.metric_threshold_name = metric_threshold_name
        self.predictions_key = predictions_key
        self.targets_key = targets_key
        self.thresholds = np.asarray(threshold, dtype=np.float32).reshape(-1)
        self.activation = instantiate_activation_block(activation) if isinstance(activation, str) else activation
        self.targets_threshold = targets_threshold

        num_thresholds = len(self.thresholds)
        self.meter = SegmentationMeter.empty(num_thresholds)
        self.beta = beta
        self.ignore_index = ignore_index

    def on_loader_start(self, runner: "IRunner"):
        self.meter.reset()

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        predictions: Tensor = (
            runner.output[self.predictions_key] if self.predictions_key is not None else runner.output
        )
        if self.activation is not None:
            predictions = self.activation(predictions)
        targets: Tensor = runner.input[self.targets_key] if self.targets_key is not None else runner.input

        predictions = torch.flatten(predictions)
        targets = torch.flatten(targets)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            predictions = predictions[mask]
            targets = targets[mask]

        thresholds = torch.from_numpy(self.thresholds).to(predictions.device).reshape(1, -1)
        predictions = predictions.view(-1, 1) >= thresholds
        targets = targets.view(-1, 1) > self.targets_threshold

        tp = (predictions & targets).sum(dim=0).float()  # [NumThresholds]
        fp = (predictions & ~targets).sum(dim=0).float()  # [NumThresholds]
        fn = (~predictions & targets).sum(dim=0).float()  # [NumThresholds]
        tn = (~predictions & ~targets).sum(dim=0).float()  # [NumThresholds]

        self.meter.tp += to_numpy(tp)
        self.meter.fp += to_numpy(fp)
        self.meter.fn += to_numpy(fn)
        self.meter.tn += to_numpy(tn)

    def on_loader_end(self, runner: "IRunner"):
        meter: SegmentationMeter = functools.reduce(lambda x, y: x + y, all_gather(self.meter))

        dice_fbeta = meter.fbeta(self.beta)

        fpr = meter.fp / (meter.fp + meter.tn + 1e-6)
        fnr = meter.fn / (meter.fn + meter.tp + 1e-6)

        best_dice_index = np.argmax(dice_fbeta)
        best_dice_threshold = self.thresholds[best_dice_index]
        best_dice_value = dice_fbeta[best_dice_index]

        runner.loader_metrics[self.metric_name] = float(best_dice_value)
        num_thresholds = len(self.thresholds)

        runner.loader_metrics[self.metric_threshold_name] = float(best_dice_threshold)

        if is_main_process() and num_thresholds > 1:
            try:
                summary_writer: SummaryWriter = get_tensorboard_logger(runner)

                f = plt.figure(figsize=(10, 10))
                plt.plot(self.thresholds, dice_fbeta)
                plt.xlabel("Threshold")
                plt.ylabel(f"Global Dice F{self.beta:.2f}")
                plt.grid()
                plt.title(f"Best threshold: {best_dice_threshold:.3f} | Dice: {best_dice_value:.3f}")
                plt.tight_layout()

                summary_writer.add_figure(
                    tag=self.metric_name + "/histogram",
                    figure=f,
                    global_step=runner.global_epoch,
                    close=True,
                )

                summary_writer.add_pr_curve_raw(
                    tag=self.metric_name + "/pr_curve",
                    true_negative_counts=meter.tn,
                    false_positive_counts=meter.fp,
                    false_negative_counts=meter.fn,
                    true_positive_counts=meter.tp,
                    precision=meter.precision,
                    recall=meter.recall,
                    num_thresholds=num_thresholds,
                    global_step=runner.global_epoch,
                )

                fig, ax1 = plt.subplots(figsize=(10, 10))

                color = 'tab:red'
                ax1.set_xlabel('Threshold')
                ax1.set_ylabel('False positive rate (FPR)', color=color)
                ax1.plot(self.thresholds, fpr, color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'tab:blue'
                ax2.set_ylabel('False negative rate (FNR)', color=color)  # we already handled the x-label with ax1
                ax2.plot(self.thresholds, fnr, color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                summary_writer.add_figure(
                    tag=self.metric_name + "/fnr_fpr",
                    figure=fig,
                    global_step=runner.global_epoch,
                    close=True,
                )

            except RuntimeError:
                pass
