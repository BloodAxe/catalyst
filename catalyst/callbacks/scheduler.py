import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner

__all__ = ["ISchedulerCallback", "SchedulerCallback"]


class ISchedulerCallback(Callback):
    """Scheduler callback interface, abstraction over scheduler step."""

    pass


class SchedulerCallback(ISchedulerCallback):
    def __init__(
        self,
    ):
        super().__init__(order=CallbackOrder.scheduler, node=CallbackNode.all)
