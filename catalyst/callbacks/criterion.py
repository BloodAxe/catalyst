import math
from typing import Dict, List, Union

from catalyst.callbacks.metric import IBatchMetricCallback
from catalyst.core.runner import IRunner
from torch import nn


class CriterionCallback(IBatchMetricCallback):
    """Callback for that measures loss with specified criterion."""

    def __init__(
        self,
        input_key: Union[str, List[str], Dict[str, str]] = "targets",
        output_key: Union[str, List[str], Dict[str, str]] = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        multiplier: float = 1.0,
        **metric_kwargs,
    ):
        """
        Args:
            input_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If '__all__', the whole input will be passed to the criterion
                If None, empty dict will be passed to the criterion.
            output_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If '__all__', the whole output will be passed to the criterion
                If None, empty dict will be passed to the criterion.
            prefix: prefix for metrics and output key for loss
                in ``runner.batch_metrics`` dictionary
            criterion_key: A key to take a criterion in case
                there are several of them and they are in a dictionary format.
            multiplier: scale factor for the output loss.
        """
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            **metric_kwargs,
        )
        self.criterion_key = criterion_key
        self._criterion = None

    @property
    def metric_fn(self):
        """Criterion function."""
        return self._criterion

    def on_stage_start(self, runner: "IRunner"):
        """Checks that the current stage has correct criterion.

        Args:
            runner: current runner
        """
        criterion = runner.get_attr(key="criterion", inner_key=self.criterion_key)
        assert criterion is not None
        self._criterion = criterion


def get_multiplier(training_progress, schedule, start, end):
    if schedule is None or schedule == "none":
        threshold = 0
    elif schedule == "linear_schedule":
        threshold = training_progress
    elif schedule == "exp_schedule":
        scale = 5
        threshold = math.exp((training_progress - 1) * scale)
        # [exp(-5), exp(0)] = [1e-2, 1]
    elif schedule == "log_schedule":
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        threshold = 1 - math.exp((-training_progress) * scale)
    else:
        raise KeyError(schedule)

    return threshold * (end - start) + start


class WeightsRegularizationCallback(CriterionCallback):
    """
    Generalized L1/L2 weight decay callback that may exponentially grow
    """

    def __init__(
        self,
        on_train_only=True,
        apply_to_bias=False,
        prefix: str = None,
        p=1,
        start_wd=0,
        end_wd=1e-4,
        schedule="exp_schedule",
    ):
        """

        :param on_train_only:
        :param apply_to_bias:
        :param prefix:
        :param p:
        :param start_wd:
        :param end_wd:
        :param schedule:
        """
        if prefix is None:
            prefix = f"l{self.p}_loss"

        super().__init__(prefix=prefix, multiplier=start_wd)

        self.on_train_only = on_train_only
        self.is_needed = True
        self.apply_to_bias = apply_to_bias
        self.schedule = schedule
        self.start_wd = start_wd
        self.end_wd = end_wd
        self.p = p
        self.multiplier = None

    def on_loader_start(self, runner: IRunner):
        self.is_needed = not self.on_train_only or runner.loader_name.startswith("train")
        if self.is_needed:
            runner.loader_metrics[f"l{self.p}_weight_decay"] = self.multiplier

    def on_epoch_start(self, runner: IRunner):
        training_progress = float(runner.epoch) / float(runner.num_epochs)
        self.multiplier = get_multiplier(training_progress, self.schedule, self.start_wd, self.end_wd)

    def on_batch_end(self, runner: IRunner):
        if not self.is_needed:
            return

        lp_reg = 0

        for module in runner.model.children():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm1d, nn.BatchNorm3d)):
                continue

            for param_name, param in module.named_parameters():
                if param_name.endswith("bias") and not self.apply_to_bias:
                    continue

                if param.requires_grad:
                    lp_reg = param.norm(self.p) * self.multiplier + lp_reg

        runner.batch_metrics.update(**{self.prefix: lp_reg.item()})


__all__ = ["CriterionCallback", "WeightsRegularizationCallback"]
