import copy
import math
import copy
import numpy as np

from catalyst.callbacks.scheduler import ISchedulerCallback
from catalyst.callbacks.schedulers.functional import scale_lr_for_param_groups
from catalyst.core import CallbackOrder, CallbackNode, IRunner

__all__ = ["CosineDecaySchedulerCallback"]


def cosine_scheduler_with_warmup(
    steps: int, initial_lr: float, final_lr_fraction: float, warmup_lr_fraction: float, warmup_num_steps: int
):
    """
    Cosine decay scheduler with warmup
    """
    warmup_lr = initial_lr * warmup_lr_fraction
    final_lr = initial_lr * final_lr_fraction
    warmup_lrs = np.linspace(warmup_lr, initial_lr, num=warmup_num_steps)
    training_fraction = np.linspace(0, 1, num=steps - warmup_num_steps)

    cosine_lr_fraction = np.cos(training_fraction * math.pi) / 2 + 0.5
    cosine_lrs = (initial_lr - final_lr) * cosine_lr_fraction + final_lr

    lr_interpolation_factors = np.concatenate([warmup_lrs, cosine_lrs])
    return lr_interpolation_factors


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
        self.warmup_lr_interpolation_factors = np.linspace(warmup_lr_fraction, 1.0, num=warmup_num_steps)
        self.original_learning_rates = None

    def __repr__(self):
        main_desc = f"Cosine decay to {self.final_lr_fraction}x of initial LR."
        if self.warmup_num_steps:
            main_desc += f" Warmup from {self.warmup_lr_fraction}x LR for {self.warmup_num_steps} steps."
        return main_desc

    def on_stage_start(self, runner: IRunner):
        self.original_learning_rates = copy.deepcopy([pg["lr"] for pg in runner.optimizer.param_groups])
        self.total_training_steps = len(runner.loaders["train"]) * runner.num_epochs

    def on_batch_start(self, runner: IRunner):
        if not runner.is_train_loader:
            return

        if runner.global_grad_update_step < self.warmup_num_steps:
            scale = self.warmup_lr_interpolation_factors[runner.global_grad_update_step]

            for original_lr, pg in zip(self.original_learning_rates, runner.optimizer.param_groups):
                pg["lr"] = original_lr * scale
        else:
            # TODO: If gradient accumulation is used, we must account for this and multiply self.warmup_num_steps * accumulation
            training_fraction = (runner.global_train_step - self.warmup_num_steps) / (
                self.total_training_steps - self.warmup_num_steps
            )
            if training_fraction < 0 or training_fraction > 1:
                raise RuntimeError(
                    f"Detected incorrect training_fraction {training_fraction} for cosine scheduler. It must stay in range [0...1]. "
                    f"Incorrect value computed on global_batch_step {runner.global_batch_step}, global_grad_update_step {runner.global_grad_update_step}, epoch {runner.global_epoch}"
                )

            cosine_decay = math.cos(training_fraction * math.pi) / 2 + 0.5
            # Interpolate LR fraction between 1 and self.final_lr_fraction
            lr_fraction = (1 - self.final_lr_fraction) * cosine_decay + self.final_lr_fraction

            if lr_fraction < 0 or lr_fraction > 1.0:
                raise RuntimeError(
                    f"Detected incorrect lr_fraction {lr_fraction} for cosine scheduler. It must stay in range [0...1]. "
                    f"Incorrect value computed on global_batch_step {runner.global_batch_step}, global_grad_update_step {runner.global_grad_update_step}, epoch {runner.global_epoch}"
                )
            scale_lr_for_param_groups(runner.optimizer.param_groups, self.original_learning_rates, lr_fraction)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    num_steps = 1_000
    x = np.arange(num_steps)
    y = cosine_scheduler_with_warmup(
        num_steps, initial_lr=1e-3, warmup_lr_fraction=0.01, warmup_num_steps=200, final_lr_fraction=0.05
    )

    plt.figure()
    plt.plot(x, y)
    plt.show()
