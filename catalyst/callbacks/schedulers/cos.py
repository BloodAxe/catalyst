import math

import numpy as np

from catalyst.callbacks.scheduler import ISchedulerCallback
from catalyst.callbacks.schedulers.functional import scale_lr_for_param_groups
from catalyst.core import CallbackOrder, CallbackNode, IRunner

__all__ = ["CosineDecaySchedulerCallback"]


class CosineDecaySchedulerCallback(ISchedulerCallback):
    total_training_steps: int

    def __init__(
        self,
        warmup_num_steps: int = 0,
        warmup_lr_fraction: float = 0.01,
        final_lr_fraction: float = 0.2,
    ):
        super().__init__(order=CallbackOrder.scheduler, node=CallbackNode.all)
        self.final_lr_fraction = final_lr_fraction
        
        self.warmup_num_steps = warmup_num_steps
        self.warmup_lr_fraction = warmup_lr_fraction
        self.warmup_lr_interpolation_factors = np.linspace(
            warmup_lr_fraction, 1.0, num=warmup_num_steps
        )
        self.original_learning_rates = None

    def __repr__(self):
        main_desc = f"Cosine decay to {self.final_lr_fraction}x of initial LR"
        if self.warmup_num_steps:
            main_desc += f" Warmup from {self.final_lr_fraction}x LR for {self.warmup_num_steps} steps."
        return main_desc
    
    def on_stage_start(self, runner: IRunner):
        self.original_learning_rates = [
            pg["lr"] for pg in runner.optimizer.param_groups
        ]

        self.total_training_steps = len(runner.loaders["train"]) * runner.num_epochs

    def on_batch_start(self, runner: IRunner):
        if not runner.is_train_loader:
            return

        if runner.global_grad_update_step < self.warmup_num_steps:
            scale = self.warmup_lr_interpolation_factors[runner.global_grad_update_step]

            for original_lr, pg in zip(
                self.original_learning_rates, runner.optimizer.param_groups
            ):
                pg["lr"] = original_lr * scale
        else:
            training_fraction = runner.global_batch_step / self.total_training_steps
            scale = math.cos(training_fraction * math.pi / 2)
            lr_scale = scale * 1.0 + (1 - scale) * self.final_lr_fraction

            scale_lr_for_param_groups(
                runner.optimizer.param_groups, self.original_learning_rates, lr_scale
            )
