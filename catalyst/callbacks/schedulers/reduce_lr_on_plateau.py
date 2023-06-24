import numpy as np

from catalyst.callbacks.scheduler import ISchedulerCallback
from catalyst.callbacks.schedulers.functional import scale_lr_for_param_groups
from catalyst.core import CallbackOrder, IRunner

__all__ = ["ReduceLROnPlateauCallback"]


class ReduceLROnPlateauCallback(ISchedulerCallback):
    warmup_num_steps: int
    warmup_lr_fraction: float

    def __init__(
        self,
        patience: int,
        multiplier: float,
        metric_to_monitor: str,
        minimize: bool,
        min_delta: float,
        warmup_num_steps: int = 0,
        warmup_lr_fraction: float = 0.01,
        min_learning_rate_fraction: float = 1e-7,
    ):
        super().__init__(order=CallbackOrder.Scheduler)
        self.patience = patience
        self.multiplier = multiplier
        self.metric_to_monitor = metric_to_monitor
        self.minimize = bool(minimize)
        self.epochs_without_improvement = 0
        self.best_value = None

        self.warmup_num_steps = warmup_num_steps
        self.warmup_lr_fraction = warmup_lr_fraction
        self.warmup_lr_interpolation_factors = np.linspace(warmup_lr_fraction, 1.0, num=warmup_num_steps)
        self.original_learning_rates = None
        self.min_learning_rate_fraction = min_learning_rate_fraction
        self.current_learning_rate_fraction = 1.0

        if minimize:
            self.is_better = lambda score, best: score <= (best - min_delta)
        else:
            self.is_better = lambda score, best: score >= (best + min_delta)

    def __repr__(self):
        main_desc = f"ReduceLROnPlateau(patience={self.patience}, factor={self.multiplier}, warmup={self.warmup_num_steps},{self.warmup_lr_fraction})"
        return main_desc

    def on_stage_start(self, runner: IRunner):
        self.original_learning_rates = [pg["lr"] for pg in runner.optimizer.param_groups]

        self.epochs_without_improvement = 0
        self.best_value = None

    def on_batch_start(self, runner: IRunner):
        if not runner.is_train_loader:
            return

        if runner.global_grad_update_step < self.warmup_num_steps:
            scale = self.warmup_lr_interpolation_factors[runner.global_grad_update_step]
            scale_lr_for_param_groups(runner.optimizer.param_groups, self.original_learning_rates, scale)

    def on_epoch_start(self, runner: IRunner):
        if self.epochs_without_improvement >= self.patience:
            self.epochs_without_improvement = 0
            self.current_learning_rate_fraction = max(
                self.min_learning_rate_fraction, self.current_learning_rate_fraction * self.multiplier
            )
            scale_lr_for_param_groups(
                runner.optimizer.param_groups, self.original_learning_rates, self.current_learning_rate_fraction
            )

    def on_epoch_end(self, runner: IRunner):
        value = runner.valid_metrics[self.metric_to_monitor]
        if self.best_value is None or self.is_better(value, self.best_value):
            self.epochs_without_improvement = 0
            self.best_value = value
        else:
            self.epochs_without_improvement += 1

    def on_stage_end(self, runner: IRunner):
        self.best_value = None
