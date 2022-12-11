import collections
import typing

import numpy as np
import torch
from torch import Tensor, nn

from catalyst.callbacks.optimizer import IOptimizerCallback
from catalyst.core import IRunner, Callback, CallbackOrder

__all__ = ["ExponentialMovingAverage", "EMACallback", "ExpEMADecay", "BetaDecay"]


class EMADecay:
    def __call__(self, step: int, total_steps: int):
        raise NotImplementedError

class ThresholdDecay(EMADecay):
    def __init__(self, decay:float):
        self.decay = decay

    def __call__(self, step: int, total_steps: int):
        decay = (step + 1) / (step + 1000)
        return np.minimum(self.decay, decay)

    def __repr__(self):
        return f"ThresholdDecay(decay={self.decay})"

class ExpEMADecay(EMADecay):
    def __init__(self, decay, beta):
        self.decay = decay
        self.beta = beta

    def __call__(self, step: int, total_steps: int):
        p = step / total_steps
        return self.decay * (1 - np.exp(-p * self.beta))


class BetaDecay(EMADecay):
    def __init__(self, beta):
        self.beta = beta

    def __repr__(self):
        return f"BetaDecay(beta={self.beta})"

    def __call__(self, step: int, total_steps: int):
        p = step / total_steps
        decay = 1 - np.exp(-p) ** self.beta
        return decay


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.

    Partially based on: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
    """

    def __init__(
        self,
        parameters: typing.Iterator[typing.Tuple[str, nn.Parameter]],
    ):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
        """
        self.ema_params = collections.OrderedDict([(k, p.clone().detach()) for k, p in parameters if p.requires_grad])

    @torch.no_grad()
    def update(self, parameters: collections.OrderedDict, ema_value: float):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """

        parameters = collections.OrderedDict([(k, p.clone().detach()) for k, p in parameters if p.requires_grad])

        if parameters.keys() != self.ema_params.keys():
            raise RuntimeError("Keys in EMA model and current model does not match")

        for key in self.ema_params.keys():
            self.ema_params[key].copy_(self.weighted_sum(self.ema_params[key], parameters[key].detach(), ema_value))

    def copy_to(self, parameters: typing.Iterator[typing.Tuple[str, nn.Parameter]]):
        """
        Copies current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for key, recipient_param in parameters:
            recipient_param.data.copy_(self.ema_params[key])

    @classmethod
    def weighted_sum(cls, averaged_weights: Tensor, current_weights: Tensor, p: float) -> Tensor:
        """
        Perform weighted sum of averaged weights and current weights using formula:
        >>> new_weights = p * averaged_weights + (1 - p) * current_weights
        """
        return p * averaged_weights + (1.0 - p) * current_weights


class EMACallback(Callback):
    """EMA weights averaging callback.
    It updates EMA weights after end of each training batch.
    On validation epoch this callback changes the model for evaluation to EMA-averaged and flip it back to "regular"
    model on start of training epochs.
    """

    def __repr__(self):
        return f"EMACallback(decay={self.decay})"

    def __init__(
        self,
        decay: EMADecay,
    ):
        super().__init__(CallbackOrder.Optimizer + 1)
        self.ema: ExponentialMovingAverage = None
        self.decay = decay
        self.non_ema_state_dict = None
        self.total_grad_update_steps = 0

    def on_stage_start(self, runner: IRunner):
        optimizer_callback: IOptimizerCallback = runner.get_callback(IOptimizerCallback)
        self.total_grad_update_steps = (
            len(runner.loaders["train"]) * runner.num_epochs
        ) // optimizer_callback.grad_accumulation_steps

        self.ema = ExponentialMovingAverage(
            parameters=runner.model.named_parameters(),
        )
        self.non_ema_state_dict = None

    def on_stage_end(self, runner: "IRunner"):
        self.ema = None
        self.non_ema_state_dict = None

    def on_loader_start(self, runner: "IRunner"):
        if runner.is_train_loader:
            # On the start of train loader we load model state dict.
            # non_ema_state_dict may be None on the first epoch
            if self.non_ema_state_dict:
                runner.model.load_state_dict(self.non_ema_state_dict)
        elif runner.is_valid_loader:
            self.non_ema_state_dict = runner.model.state_dict()
            self.ema.copy_to(runner.model.named_parameters())

    def on_loader_end(self, runner: IRunner):
        pass

    def on_grad_step_end(self, runner: IRunner):
        if not runner.is_train_loader:
            raise RuntimeError(
                "A on_grad_step_end called from non-train loader. " "This is likey a bug in the library"
            )

        decay = self.decay(
            step=runner.global_grad_update_step,
            total_steps=self.total_grad_update_steps,
        )

        runner.batch_metrics["_ema/decay"] = decay

        self.ema.update(runner.model.named_parameters(), decay)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    total_steps = 1000000
    steps = np.linspace(1, total_steps, total_steps, endpoint=True)

    plt.figure()

    for name, ema_algs in [
        ("beta", BetaDecay(beta=15)),
        ("threshold",ThresholdDecay(0.9998)),
        ("exp", ExpEMADecay(decay=0.9998,beta=4))
    ]:
        plt.plot(steps, ema_algs(steps, total_steps), label=name)

    plt.legend()
    plt.tight_layout()
    plt.show()