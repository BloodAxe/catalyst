from catalyst.callbacks.scheduler import ISchedulerCallback
from catalyst.core import CallbackOrder

__all__ = ["ReduceLROnPlateauCallback"]


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
