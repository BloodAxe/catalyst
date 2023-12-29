import logging
from typing import Union, List, Callable, Optional, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_toolbelt.modules import instantiate_activation_block
from pytorch_toolbelt.utils import all_gather, to_numpy, is_main_process, reduce_dict_sum
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from catalyst.callbacks.metrics.segmentation_utils import SegmentationMeter
from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.utils import get_tensorboard_logger

logger = logging.getLogger("catalyst.callbacks.metrics.BinaryDiceScore")


class BinaryDiceScore(Callback):
    """
    Metric callback to compute binary dice score per scene.
    This callback supports following features:
    - Computation of F-beta dice score metric (Default: beta = 1.0 )
    - Threshold tuning by passing a list of thresholds (Default: 0.5)
    - Logging of the plot of dice score (Y axis) vs threshold value (X axis)
    - Ignoring specific targets during metric computation
    - Computation of metric per scene and averaging over all scenes
    - Custom activation function for outputs (Default: sigmoid)
    """

    def __init__(
        self,
        predictions_key: Union[str, int, None],
        targets_key: Union[str, int, None],
        scene_key: Union[str, int, None] = None,
        activation: Union[None, str, Callable[[Tensor], Tensor], nn.Module] = torch.sigmoid,
        threshold: Union[float, List[float], np.ndarray] = 0.5,
        metric_name: str = "metrics/mean_dice",
        metric_threshold_name="metrics/mean_dice_threshold",
        beta: float = 1.0,
        ignore_index: Optional[int] = None,
        targets_threshold: float = 0.5,
    ):
        """
        :param predictions_key: name of the key in ``runner.output`` dictionary with predictions
        :param targets_key: name of the key in ``runner.input`` dictionary with targets
        :param scene_key: name of the key in ``runner.input`` dictionary with scene ids. If None, each sample in batch considered as a unique scene
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
        self.scene_key = scene_key
        self.activation = instantiate_activation_block(activation) if isinstance(activation, str) else activation
        self.targets_threshold = targets_threshold

        self.num_thresholds = len(self.thresholds)
        self.per_scene_meters = {}
        self.beta = beta
        self.ignore_index = ignore_index

    def on_loader_start(self, runner: "IRunner"):
        self.per_scene_meters = {}

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        scenes = runner.input[self.scene_key] if self.scene_key is not None else None
        predictions: Tensor = (
            runner.output[self.predictions_key] if self.predictions_key is not None else runner.output
        )
        if self.activation is not None:
            predictions = self.activation(predictions)
        targets: Tensor = runner.input[self.targets_key] if self.targets_key is not None else runner.input

        thresholds = (
            torch.from_numpy(self.thresholds).to(device=predictions.device, dtype=predictions.dtype).reshape(1, -1)
        )

        for scene_id, prediction, target in zip(scenes, predictions, targets):
            if self.ignore_index is not None:
                target = torch.flatten(target)
                prediction = torch.flatten(prediction)

                mask = target != self.ignore_index
                prediction = prediction[mask]
                target = target[mask]

                if len(target) == 0:
                    continue

            prediction = prediction.view(-1, 1) >= thresholds
            target = target.view(-1, 1) > self.targets_threshold

            tp = to_numpy((prediction & target).sum(dim=0).float())  # [NumThresholds]
            fp = to_numpy((prediction & ~target).sum(dim=0).float())  # [NumThresholds]
            fn = to_numpy((~prediction & target).sum(dim=0).float())  # [NumThresholds]
            tn = to_numpy((~prediction & ~target).sum(dim=0).float())  # [NumThresholds]

            if scene_id not in self.per_scene_meters:
                self.per_scene_meters[scene_id] = SegmentationMeter.empty(self.num_thresholds)

            meter = self.per_scene_meters[scene_id]
            meter.tp += tp
            meter.fp += fp
            meter.fn += fn
            meter.tn += tn

    def on_loader_end(self, runner: "IRunner"):
        all_scenes: Mapping[str, SegmentationMeter] = reduce_dict_sum(self.per_scene_meters)

        scene_names = list(all_scenes.keys())
        dice_fbeta = np.stack(
            [all_scenes[scene_name].fbeta(self.beta) for scene_name in scene_names], axis=0
        )  # [NumScenes, NumThresholds]
        mean_dice_fbeta = np.mean(dice_fbeta, axis=0)  # [NumThresholds]

        best_dice_index = np.argmax(mean_dice_fbeta)
        best_dice_threshold = self.thresholds[best_dice_index]
        best_dice_value = mean_dice_fbeta[best_dice_index]

        runner.loader_metrics[self.metric_name] = float(best_dice_value)
        num_thresholds = len(self.thresholds)

        runner.loader_metrics[self.metric_threshold_name] = float(best_dice_threshold)

        if is_main_process() and num_thresholds > 1:
            try:
                summary_writer: SummaryWriter = get_tensorboard_logger(runner)
                f = plt.figure(figsize=(16, 16))
                plt.plot(self.thresholds, mean_dice_fbeta, label="Average", linewidth=3, color="red")
                plt.xlabel("Threshold")
                plt.ylabel(f"Dice F{self.beta:.2f} (Averaged per scene)")
                plt.grid()

                for scene_name, dice_fbeta in zip(scene_names, dice_fbeta):
                    plt.plot(self.thresholds, dice_fbeta, alpha=0.5, linewidth=2, label=scene_name)

                plt.title(f"Best threshold: {best_dice_threshold:.3f} | Dice: {best_dice_value:.3f}")
                plt.legend()
                plt.tight_layout()

                summary_writer.add_figure(
                    tag=self.metric_name + "/histogram",
                    figure=f,
                    global_step=runner.global_epoch,
                )

            except Exception:
                logger.error("Can't log visualization for global binary dice score", exc_info=True)
