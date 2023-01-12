import copy
from typing import Union

import numpy as np

from catalyst.callbacks import ISchedulerCallback
from catalyst.callbacks.schedulers.functional import scale_lr_for_param_groups
from catalyst.core import CallbackOrder, CallbackNode, IRunner

__all__ = ["CyclicLRSchedulerCallback"]

def one_cycle_schedule(
    steps: int,
    min_lr: float,
    max_lr: float,
    warmup_fraction: float,
    warmup_pattern: str = "linear",
    cooldown_pattern: str = "cosine",
):
    warmup_steps = int(steps * warmup_fraction)
    cooldown_steps = steps - warmup_steps

    if warmup_pattern == "linear":
        warmup = np.linspace(min_lr, max_lr, num=warmup_steps, endpoint=True)
    elif warmup_pattern == "cosine":
        warmup = ((np.cos(np.linspace(np.pi, 0, num=warmup_steps, endpoint=True)) + 1) / 2) * (
            max_lr - min_lr
        ) + min_lr
    else:
        raise ValueError(f"Unknown warmup pattern: {warmup_pattern}")

    if cooldown_pattern == "cosine":
        cooldown = ((np.cos(np.linspace(0, np.pi, num=cooldown_steps, endpoint=True)) + 1) / 2) * (
            max_lr - min_lr
        ) + min_lr
    elif cooldown_pattern == "linear":
        cooldown = np.linspace(max_lr, min_lr, num=cooldown_steps, endpoint=True)
    else:
        raise ValueError("Unknown cooldown pattern: {cooldown_pattern}")

    return np.concatenate([warmup, cooldown])


def compute_clr_schedule(
    total_steps: int,
    min_lr: float,
    max_lr: float,
    num_cycles: int,
    decay_per_cycle: float,
    warmup_fraction: float,
    warmup_pattern: str = "linear",
    cooldown_pattern: str = "cosine",
):
    """
    Compute the learning rate for following a repeating pattern:
    In each cycle, learning rate linearly increase from min_lr to max_lr and
    decay via cosine rule to min_lr. The number of cycles is controlled by num_cycles.

    Args:
        current_step: The current step in the cycle.
        total_steps: The total number of steps in the cycle.
        min_lr: The minimum learning rate.
        max_lr: The maximum learning rate.
        num_cycles: The number of cycles.
    Returns:
        The learning rate for the given step.
    """
    steps = np.floor(np.linspace(0, total_steps, endpoint=True, num=num_cycles + 1)).astype(int)
    steps_in_each_cycle = steps[1:] - steps[:-1]
    total_schedule = []

    for i, step_size in enumerate(steps_in_each_cycle):
        lr_schedules = one_cycle_schedule(
            step_size,
            min_lr,
            max_lr * (decay_per_cycle**i),
            warmup_fraction=warmup_fraction,
            warmup_pattern=warmup_pattern,
            cooldown_pattern=cooldown_pattern,
        )
        total_schedule.append(lr_schedules)

    return np.concatenate(total_schedule)


class CyclicLRSchedulerCallback(ISchedulerCallback):
    def __init__(
        self,
        min_lr_fraction: float,
        num_cycles: int,
        decay_per_cycle: float,
        warmup_fraction: float,
        warmup_pattern: str = "linear",
        cooldown_pattern: str = "cosine",
    ):
        super().__init__(order=CallbackOrder.Scheduler, node=CallbackNode.All)

        self.min_lr = min_lr_fraction
        self.max_lr = 1.0

        self.num_cycles = num_cycles
        self.decay_per_cycle = decay_per_cycle

        self.warmup_fraction = warmup_fraction
        self.warmup_pattern = warmup_pattern
        self.cooldown_pattern = cooldown_pattern

    def on_stage_start(self, runner: IRunner):
        self.original_learning_rates = copy.deepcopy([pg["lr"] for pg in runner.optimizer.param_groups])
        self.total_training_steps = len(runner.loaders["train"]) * runner.num_epochs
        self.learning_rates = compute_clr_schedule(
            min_lr=self.min_lr,
            max_lr=self.max_lr,
            warmup_fraction=self.warmup_fraction,
            decay_per_cycle=self.decay_per_cycle,
            warmup_pattern=self.warmup_pattern,
            cooldown_pattern=self.cooldown_pattern,
            num_cycles=self.num_cycles,
            total_steps=self.total_training_steps,
        )
    def on_batch_start(self, runner: IRunner):
        if not runner.is_train_loader:
            return

        fraction = self.learning_rates[runner.global_grad_update_step]
        scale_lr_for_param_groups(runner.optimizer.param_groups, self.original_learning_rates, fraction)


if __name__ == "__main__":
    num_steps = 10000
    num_cycles = 17
    x = np.arange(num_steps)
    y = compute_clr_schedule(
        num_steps,
        min_lr=1e-6,
        max_lr=1e-3,
        warmup_fraction=0.24,
        decay_per_cycle=0.9,
        warmup_pattern="linear",
        cooldown_pattern="cosine",
        num_cycles=num_cycles,
        total_steps=num_steps,
    )

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x, y)
    plt.show()
