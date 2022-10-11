import logging
import warnings
from typing import Callable, Dict, List, TYPE_CHECKING, Mapping

import hydra.utils
import torch
from torch import nn
from torch.distributed.optim import ZeroRedundancyOptimizer

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


def grad_norm(model: nn.Module, prefix: str, norm_type: int) -> Dict[str, float]:
    """Computes gradient norms for a given model.

    Args:
        model: model which gradients to be saved.
        prefix: prefix for keys in resulting dictionary.
        norm_type: norm type of gradient norm.

    Returns:
        Dict: dictionary in which gradient norms are stored.
    """
    from torch.nn import DataParallel
    from torch.nn.parallel import DistributedDataParallel

    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module

    total_norm = 0.0
    grad_norm = {}

    for tag, value in model.named_parameters():
        tag = tag.replace(".", "/")
        metrics_tag = f"{prefix}/{tag}"
        param_norm = value.grad.data.norm(norm_type).item()
        total_norm += param_norm**norm_type
        grad_norm[metrics_tag] = param_norm

    total_norm = total_norm ** (1.0 / norm_type)
    metrics_tag = f"{prefix}/total"
    grad_norm[metrics_tag] = total_norm

    return grad_norm


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
        log_grad_norm: bool = False,
        grad_norm_type: int = 2,
        grad_norm_prefix: str = "_grad_norm",
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

        self.grad_clip_params = grad_clip_params

        self._optimizer_step_fn: Callable = None
        self.use_fast_zero_grad = use_fast_zero_grad

        self.log_grad_norm = log_grad_norm
        self.grad_norm_prefix = grad_norm_prefix
        self.grad_norm_type = grad_norm_type

    def _optimizer_step(self, optimizer: Optimizer) -> None:
        """CPU and GPU optimization step.

        Args:
            optimizer: optimizer object
        """
        optimizer.step()

    def grad_step(
        self,
        runner,
        optimizer: Optimizer,
        grad_clip_params: Mapping = None,
    ) -> None:
        """Makes a gradient step for a given optimizer.

        Args:
            optimizer: the optimizer
            grad_clip_fn: function for gradient clipping
        """

        # Clip
        if grad_clip_params is not None:
            for group in optimizer.param_groups:
                parameters = group["params"]
                torch.nn.utils.clip_grad_norm_(parameters, **grad_clip_params)

        # Log
        if self.log_grad_norm:
            grad_norm_dict = grad_norm(runner.model, self.grad_norm_prefix, self.grad_norm_type)
            runner.batch_metrics.update(**grad_norm_dict)

        # Step
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
                runner,
                optimizer=self._optimizer,
                grad_clip_params=self.grad_clip_params,
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

        if isinstance(self._optimizer, ZeroRedundancyOptimizer):
            self._optimizer.consolidate_state_dict()

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
        log_grad_norm: bool = False,
        grad_norm_type: int = 2,
        grad_norm_prefix: str = "_grad_norm",
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

        self.grad_clip_params = grad_clip_params

        # Initialized at on_state_start()
        self.scaler = None

        self.log_grad_norm = log_grad_norm
        self.grad_norm_type = grad_norm_type
        self.grad_norm_prefix = grad_norm_prefix

    def grad_step(
        self,
        runner,
        optimizer: Optimizer,
        grad_clip_params=None,
    ) -> None:
        """Makes a gradient step for a given optimizer.

        Args:
            optimizer: the optimizer
            grad_clip_fn: function for gradient clipping
        """
        if grad_clip_params is not None or self.log_grad_norm:
            # Unscales the gradients of
            # optimizer's assigned params in-place
            self.scaler.unscale_(optimizer)

            if grad_clip_params is not None:
                for group in optimizer.param_groups:
                    # Since the gradients of optimizer's
                    # assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(group["params"], **grad_clip_params)

            if self.log_grad_norm:
                grad_norm_dict = grad_norm(runner.model, self.grad_norm_prefix, self.grad_norm_type)
                runner.batch_metrics.update(**grad_norm_dict)

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
                runner,
                optimizer=self._optimizer,
                grad_clip_params=self.grad_clip_params,
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
        if isinstance(self._optimizer, ZeroRedundancyOptimizer):
            self._optimizer.consolidate_state_dict()

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
