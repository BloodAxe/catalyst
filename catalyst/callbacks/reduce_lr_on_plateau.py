import numpy as np

from catalyst.callbacks import ControlFlowCallback
from catalyst.callbacks.scheduler import ISchedulerCallback
from catalyst.core import CallbackOrder

__all__ = ["ReduceLROnPlateauCallback"]


class WarmupLRSchedulerCallback(ISchedulerCallback):
    def __init__(self, num_steps: int, warmup_lr_fraction: float):
        super().__init__(order=CallbackOrder.Scheduler)
        self.num_steps = num_steps
        self.warmup_lr_fraction = warmup_lr_fraction
        self.original_learning_rates = None
        self.lr_factors = np.linspace(self.warmup_lr_fraction, 1.0, num=self.num_steps)
    
    def has_finite_number_of_epochs(self):
        return False
    def on_stage_start(self, runner: "IRunner"):
        self.original_learning_rates = [
            pg["lr"] for pg in runner.optimizer.param_groups
        ]

    def on_batch_start(self, runner: "IRunner"):
        if runner.is_train_loader and runner.global_batch_step <= self.num_steps:
            alpha = self.lr_factors[runner.global_batch_step]

            for original_lr, pg in zip(
                self.original_learning_rates, runner.optimizer.param_groups
            ):
                pg["lr"] = original_lr * alpha


class ReduceLROnPlateauCallback(ISchedulerCallback):
    def __init__(
        self,
        patience: int,
        multiplier: float,
        metric_to_monitor: str,
        minimize: bool,
        min_delta: float,
    ):
        super().__init__(order=CallbackOrder.Scheduler)
        self.patience = patience
        self.multiplier = multiplier
        self.metric_to_monitor = metric_to_monitor
        self.minimize = bool(minimize)
        self.epochs_without_improvement = 0
        self.best_value = None

        if minimize:
            self.is_better = lambda score, best: score <= (best - min_delta)
        else:
            self.is_better = lambda score, best: score >= (best + min_delta)

    def on_stage_start(self, runner: "IRunner"):
        self.epochs_without_improvement = 0
        self.best_value = None

    def on_epoch_start(self, runner: "IRunner"):
        if self.epochs_without_improvement >= self.patience:
            for pg in runner.optimizer.param_groups:
                pg["lr"] *= self.multiplier

    def on_epoch_end(self, runner: "IRunner"):
        value = runner.valid_metrics[self.metric_to_monitor]
        best_value = self.best_value
        if self.best_value is None or self.is_better(value, best_value):
            self.epochs_without_improvement = 0
            self.best_value = value
        else:
            self.epochs_without_improvement += 1

    def on_stage_end(self, runner: "IRunner"):
        self.best_value = None
