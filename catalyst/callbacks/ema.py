import collections
import typing

import numpy as np
import torch
from catalyst.callbacks.optimizer import IOptimizerCallback
from catalyst.core import IRunner, Callback, CallbackOrder
from torch import Tensor, nn

__all__ = ["ExponentialMovingAverage", "EMACallback", "ExpEMADecay", "BetaDecay", "ThresholdDecay"]


class EMADecay:
    def __call__(self, step: int, total_steps: int):
        raise NotImplementedError


class ThresholdDecay(EMADecay):
    def __init__(self, decay: float):
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
        self.averaged_params = collections.OrderedDict(
            [(k, p.clone().detach()) for k, p in parameters if p.requires_grad]
        )

    @torch.no_grad()
    def update(self, parameters: typing.Iterator[typing.Tuple[str, nn.Parameter]], ema_fraction: float):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                    parameters used to initialize this object.
        ema_fraction: Weight factor for EMA weights. Model weights get weight 1 - ema_fraction
        """

        parameters = collections.OrderedDict([(k, p.detach().clone()) for k, p in parameters if p.requires_grad])

        if parameters.keys() != self.averaged_params.keys():
            raise RuntimeError("Keys in EMA model and current model does not match")

        for key in self.averaged_params.keys():
            self.averaged_params[key].copy_(
                self.weighted_sum(self.averaged_params[key], parameters[key], ema_fraction)
            )

    def copy_to(self, parameters: typing.Iterator[typing.Tuple[str, nn.Parameter]]):
        """
        Copies current EMA parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for key, recipient_param in parameters:
            recipient_param.data.copy_(self.averaged_params[key])

    @classmethod
    def weighted_sum(cls, averaged_weights: Tensor, current_weights: Tensor, p: float) -> Tensor:
        """
        Perform weighted sum of averaged weights and current weights using formula:
        >>> new_weights = p * averaged_weights + (1 - p) * current_weights
        """
        return p * averaged_weights + (1.0 - p) * current_weights


class EMACallback(Callback):
    """Weight averaging callback for exponential weight averaging.
    On start of training it saves down model params to an in internal storage and update it's weights using EMA rule
    after each gradient step.
    On start of train loader it loads weights of regular model back.
    On start of validation loader it replaces the model state with EMA weights (That is on validation you're getting EMA-model performance metric).
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

        if len(runner.loaders["train"]) % optimizer_callback.grad_accumulation_steps != 0:
            self.get_callback_logger().warning(
                "Length of train loader contains non-integer number of gradient updates. "
                "Last batch would not contribute to grad update."
            )
        self.ema = ExponentialMovingAverage(
            parameters=runner.model.named_parameters(),
        )
        self.non_ema_state_dict = None

    def _on_train_loader_start(self, runner: "IRunner"):
        """
        On the start of train epoch we restore the saved (non-EMA) model state dict.
        A non_ema_state_dict is None on the first epoch.
        """
        if self.non_ema_state_dict is not None:
            runner.model.load_state_dict(self.non_ema_state_dict)

    def on_grad_step_end(self, runner: IRunner):
        if not runner.is_train_loader:
            raise RuntimeError(
                "A on_grad_step_end called from non-train loader. This is probably a bug in the library"
            )

        decay = self.decay(
            step=runner.global_grad_update_step,
            total_steps=self.total_grad_update_steps,
        )

        runner.batch_metrics["_ema/decay"] = decay
        self.ema.update(runner.model.named_parameters(), decay)

    def _on_train_loader_end(self, runner: "IRunner"):
        """
        Save the non-EMA model state to internal state as it will be flipped to EMA version on validation.
        """
        self.non_ema_state_dict = runner.model.state_dict()

    def _on_valid_loader_start(self, runner: "IRunner"):
        """
        On start of validation we load the weights of EMA model
        :param runner:
        :return:
        """
        self.ema.copy_to(runner.model.named_parameters())

    def _on_valid_loader_end(self, runner: "IRunner"):
        """
        We do nothing on end of validation to ensure that saved checkpoints will be written with EMA weights.
        Only on the start of new train epoch we may restore the weights of original model.
        Note this would probably break the resume training functionality. Ideally we should save both ema and non-ema weights.
        TODO: Implement state_dict() method for all callbacks to allow saving state of each callback.

        :param runner:
        :return:
        """
        pass

    def on_stage_end(self, runner: "IRunner"):
        self.ema.copy_to(runner.model.named_parameters())
        self.ema = None
        self.non_ema_state_dict = None

    def on_loader_start(self, runner: "IRunner"):
        if runner.is_train_loader:
            self._on_train_loader_start(runner)
        elif runner.is_valid_loader:
            self._on_valid_loader_start(runner)

    def on_loader_end(self, runner: IRunner):
        if runner.is_train_loader:
            self._on_train_loader_end(runner)
        elif runner.is_valid_loader:
            self._on_valid_loader_end(runner)
