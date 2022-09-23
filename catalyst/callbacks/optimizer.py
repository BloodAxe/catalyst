import logging
import warnings
from typing import Callable, Dict, List, TYPE_CHECKING

import hydra.utils
import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.typing import Optimizer
from catalyst.utils.misc import maybe_recursive_call
from catalyst.utils.torch import get_optimizer_momentum

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner

logger = logging.getLogger(__name__)


def zero_grad(optimizer: Optimizer) -> None:
    """Perform an hacky way to zero gradients.

    Args:
        optimizer: optimizer with model parameters.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = None


class IOptimizerCallback(Callback):
    """Optimizer callback interface, abstraction over optimizer step."""

    pass


class OptimizerCallback(IOptimizerCallback):
    """Optimizer callback, abstraction over optimizer step."""

    def __init__(
        self,
        metric_key: str = None,
        optimizer_key: str = None,
        accumulation_steps: int = 1,
        grad_clip_params: Dict = None,
        loss_key: str = None,
        use_fast_zero_grad: bool = True,
    ):
        """
        Args:
            loss_key: key to get loss from ``runner.batch_metrics``
            optimizer_key: A key to take a optimizer in case
                there are several of them and they are in a dictionary format.
            accumulation_steps: number of steps before
                ``model.zero_grad()``
            grad_clip_params: params for gradient clipping
            use_fast_zero_grad: boost ``optimizer.zero_grad()``,
                default is ``False``.
        """
        super().__init__(order=CallbackOrder.optimizer, node=CallbackNode.all)
        assert metric_key is None or loss_key is None
        if loss_key is not None:
            warnings.warn(
                "OptimizerCallback: " "`loss_key` is now deprecated in favor `metric_key`",
                stacklevel=2,
            )
        self.metric_key: str = metric_key or loss_key or "loss"
        self.optimizer_key: str = optimizer_key

        self.accumulation_steps: int = accumulation_steps
        self._accumulation_counter: int = 0

        self.grad_clip_fn = hydra.utils.instantiate(**grad_clip_params) if grad_clip_params is not None else None

        self._optimizer_step_fn: Callable = None
        self.use_fast_zero_grad = use_fast_zero_grad

    def _optimizer_step(self, optimizer: Optimizer) -> None:
        """CPU and GPU optimization step.

        Args:
            optimizer: optimizer object
        """
        optimizer.step()

    def grad_step(
        self,
        *,
        optimizer: Optimizer,
        optimizer_wds: List[float] = 0,
        grad_clip_fn: Callable = None,
    ) -> None:
        """Makes a gradient step for a given optimizer.

        Args:
            optimizer: the optimizer
            optimizer_wds: list of weight decay parameters
                for each param group
            grad_clip_fn: function for gradient clipping
        """
        for group in zip(optimizer.param_groups):
            if grad_clip_fn is not None:
                grad_clip_fn(group["params"])
        # optimize parameters
        self._optimizer_step_fn(optimizer)

    def on_stage_start(self, runner: "IRunner") -> None:
        """Checks that the current stage has correct optimizer.

        Args:
            runner(IRunner): current runner
        """
        self._optimizer = runner.get_attr(key="optimizer", inner_key=self.optimizer_key)
        # device based optimization step
        self._optimizer_step_fn = self._optimizer_step

        assert self._optimizer is not None

    def on_loader_start(self, runner: "IRunner"):
        self._accumulation_counter = 0

    def on_batch_end(self, runner: "IRunner") -> None:
        """On batch end event

        Args:
            runner: current runner
        """
        if not runner.is_train_loader:
            return

        loss = runner.batch_metrics[self.metric_key]

        self._accumulation_counter += 1
        need_gradient_step = self._accumulation_counter % self.accumulation_steps == 0

        loss.backward()

        if need_gradient_step:
            self.grad_step(
                optimizer=self._optimizer,
                grad_clip_fn=self.grad_clip_fn,
            )
            if not self.use_fast_zero_grad:
                maybe_recursive_call(self._optimizer, "zero_grad")
            else:
                maybe_recursive_call(self._optimizer, zero_grad)
            self._accumulation_counter = 0

    def on_epoch_end(self, runner: "IRunner") -> None:
        """On epoch end event.

        Args:
            runner: current runner
        """

        lr = self._optimizer.param_groups[0]["lr"]
        lr_name = f"lr/{self.optimizer_key}" if self.optimizer_key is not None else "lr"
        runner.epoch_metrics[lr_name] = lr

        momentum = get_optimizer_momentum(self._optimizer)
        if momentum is not None:
            momentum_name = f"momentum/{self.optimizer_key}" if self.optimizer_key is not None else "momentum"
            runner.epoch_metrics[momentum_name] = momentum


class AMPOptimizerCallback(IOptimizerCallback):
    """
    Optimizer callback with native torch amp support.
    """

    def __init__(
        self,
        metric_key: str = None,
        optimizer_key: str = None,
        accumulation_steps: int = 1,
        grad_clip_params: Dict = None,
        loss_key: str = None,
        use_fast_zero_grad: bool = True,
    ):
        """
        Args:
            loss_key: key to get loss from ``runner.batch_metrics``
            optimizer_key: A key to take a optimizer in case
                there are several of them and they are in a dictionary format.
            accumulation_steps: number of steps before
                ``model.zero_grad()``
            grad_clip_params: params for gradient clipping
            decouple_weight_decay: If True - decouple weight decay
                regularization.
        """
        super().__init__(order=CallbackOrder.optimizer, node=CallbackNode.all)
        assert metric_key is None or loss_key is None
        if loss_key is not None:
            warnings.warn(
                "OptimizerCallback: " "`loss_key` is now deprecated in favor `metric_key`",
                stacklevel=2,
            )
        self.metric_key: str = metric_key or loss_key or "loss"
        self.optimizer_key: str = optimizer_key

        self.accumulation_steps: int = accumulation_steps
        self._accumulation_counter: int = 0
        self.use_fast_zero_grad = use_fast_zero_grad

        self.grad_clip_fn = hydra.utils.instantiate(**grad_clip_params) if grad_clip_params is not None else None

        # Initialized at on_state_start()
        self.scaler = None

    def grad_step(
        self,
        *,
        optimizer: Optimizer,
        grad_clip_fn: Callable = None,
    ) -> None:
        """Makes a gradient step for a given optimizer.

        Args:
            optimizer: the optimizer
            grad_clip_fn: function for gradient clipping
        """
        if grad_clip_fn is not None:
            # Unscales the gradients of
            # optimizer's assigned params in-place
            self.scaler.unscale_(optimizer)
            for group in optimizer.param_groups:
                # Since the gradients of optimizer's
                # assigned params are unscaled, clips as usual:
                grad_clip_fn(group["params"])

        self.scaler.step(optimizer)
        self.scaler.update()

    def on_stage_start(self, runner: "IRunner") -> None:
        """Checks that the current stage has correct optimizer.

        Args:
            runner(IRunner): current runner
        """
        from torch.cuda.amp import GradScaler

        self._optimizer = runner.get_attr(key="optimizer", inner_key=self.optimizer_key)
        self.scaler = GradScaler()
        assert self._optimizer is not None

    def on_batch_start(self, runner: "IRunner") -> None:
        """On batch start event

        Args:
            runner: current runner
        """
        self.prev_autocast_state = torch.is_autocast_enabled()
        torch.set_autocast_enabled(True)
        torch.autocast_increment_nesting()

    def on_batch_end(self, runner: "IRunner") -> None:
        """On batch end event

        Args:
            runner: current runner
        """
        # Drop the cache when we exit to a nesting level
        # that's outside any instance of autocast.
        if torch.autocast_decrement_nesting() == 0:
            torch.clear_autocast_cache()
        torch.set_autocast_enabled(self.prev_autocast_state)

        if not runner.is_train_loader:
            return

        loss = runner.batch_metrics[self.metric_key]

        self._accumulation_counter += 1
        need_gradient_step = self._accumulation_counter % self.accumulation_steps == 0

        self.scaler.scale(loss).backward()

        if need_gradient_step:
            self.grad_step(
                optimizer=self._optimizer,
                grad_clip_fn=self.grad_clip_fn,
            )
            if not self.use_fast_zero_grad:
                maybe_recursive_call(self._optimizer, "zero_grad")
            else:
                maybe_recursive_call(self._optimizer, zero_grad)
            self._accumulation_counter = 0

    def on_epoch_end(self, runner: "IRunner") -> None:
        """On epoch end event.

        Args:
            runner: current runner
        """
        lr = self._optimizer.param_groups[0]["lr"]
        lr_name = f"lr/{self.optimizer_key}" if self.optimizer_key is not None else "lr"
        runner.epoch_metrics[lr_name] = lr

        momentum = get_optimizer_momentum(self._optimizer)
        if momentum is not None:
            momentum_name = f"momentum/{self.optimizer_key}" if self.optimizer_key is not None else "momentum"
            runner.epoch_metrics[momentum_name] = momentum

    def on_stage_end(self, runner: "IRunner") -> None:
        """On stage end event.

        Args:
            runner: current runner
        """
        self.scaler = None


__all__ = [
    "IOptimizerCallback",
    "AMPOptimizerCallback",
    "OptimizerCallback",
]
