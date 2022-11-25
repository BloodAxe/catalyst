import math

import torch
from torch import Tensor

from catalyst.core import IRunner, Callback, CallbackOrder

__all__ = ["ExponentialMovingAverage", "EMABatchCallback", "EMAEpochCallback"]


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.

    Partially based on: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
    """

    def __init__(self, parameters, decay: float, beta: float = 1, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.beta = beta
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]

    @torch.no_grad()
    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))

        parameters = [p for p in parameters if p.requires_grad]
        for ema_param, model_param in zip(self.shadow_params, parameters):
            ema_param.copy_(self.weighted_sum(ema_param, model_param.detach(), decay))

    def compute_weighting_factor(self, step):
        return self.decay * (1 - math.exp(-step * self.beta))

    @classmethod
    def weighted_sum(
        cls, averaged_weights: Tensor, current_weights: Tensor, p: float
    ) -> Tensor:
        return p * averaged_weights + (1.0 - p) * current_weights

    def copy_to(self, parameters):
        """
        Copies current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)


class EMABatchCallback(Callback):
    """EMA weights averaging callback.
    On validation epoch this callback changes the model for evaluation to EMA-averaged and flip it back to "regular"
    model on training epochs.
    """

    def __repr__(self):
        return f"EMABatchCallback(decay={self.decay}, use_num_updates={self.use_num_updates})"

    def __init__(
        self,
        decay: float = 0.99,
        use_num_updates: bool = True,
        apply_after_epoch: int = 0,
    ):
        super().__init__(CallbackOrder.Optimizer + 1)
        self.ema = None
        self.decay = decay
        self.apply_after_epoch = apply_after_epoch
        self.use_num_updates = use_num_updates
        self.model_state_dict = None

    def on_stage_start(self, runner: IRunner):
        self.ema = ExponentialMovingAverage(
                runner.model.parameters(),
                decay=self.decay,
                beta=self.beta,
                use_num_updates=self.use_num_updates,
            )
        self.model_state_dict = None

    def on_loader_start(self, runner: "IRunner"):
        if runner.is_train_loader:
            pass
        elif runner.is_valid_loader:
            self.model_state_dict = runner.model.state_dict()
            self.ema.copy_to(runner.model)
        else:
            pass

    def on_batch_end(self, runner: IRunner):
        if state.is_train_loader and state.epoch >= self.apply_after_epoch:
            self.ema.update(state.model.parameters())

    def on_loader_end(self, runner:IRunner):
        if runner.is_train_loader:
            self.ema.copy_to(runner.model)
        elif runner.is_valid_loader:
            pass
        else:
            pass

        if state.is_train_loader and state.epoch >= self.apply_after_epoch:
            self.ema.copy_to(state.model.parameters())

    def on_stage_end(self, runner: "IRunner"):
        self.ema = None


class EMAEpochCallback(Callback):
    """
    This EMA callback:
    1) Saves a copy of parameters on start of the training stage
    2) On each training step end it updates the shadow copy using EMA
    3) On start of each load it loads model weights from shadow copy
    """

    def __repr__(self):
        return f"EMAEpochCallback(decay={self.decay}, apply_after_epoch={self.apply_after_epoch}, use_num_updates={self.use_num_updates})"

    def __init__(self, decay: float = 0.99, apply_after_epoch=0, use_num_updates=False):
        super().__init__(CallbackOrder.Optimizer + 1)
        self.ema = None
        self.decay = decay
        self.apply_after_epoch = apply_after_epoch
        self.use_num_updates = use_num_updates

    def on_stage_start(self, runner: IRunner):
        self.ema = ExponentialMovingAverage(
            runner.model.parameters(),
            decay=self.decay,
            use_num_updates=self.use_num_updates,
        )

    def on_loader_start(self, state: IRunner):
        self.ema.copy_to(state.model.parameters())

    def on_loader_end(self, runner):
        if runner.is_train_loader and runner.epoch > self.apply_after_epoch:
            self.ema.update(runner.model.parameters())
