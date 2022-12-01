import os
import random
from collections import OrderedDict
from typing import Union

import torch
import torch.distributed
from packaging.version import parse, Version
from torch import nn



def check_ddp_wrapped(model: nn.Module) -> bool:
    """
    Checks whether model is wrapped with DataParallel/DistributedDataParallel.
    """
    parallel_wrappers = nn.DataParallel, nn.parallel.DistributedDataParallel
    return isinstance(model, parallel_wrappers)


def check_amp_available() -> bool:
    """Checks if torch.amp is available."""
    return parse(torch.__version__) >= Version("1.6.0")


def check_torch_distributed_initialized() -> bool:
    """Checks if torch.distributed is available and initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def maybe_torch_distributed_barrier():
    if check_torch_distributed_initialized():
        torch.distributed.barrier()


def check_slurm_available():
    """Checks if slurm is available."""
    return "SLURM_JOB_NUM_NODES" in os.environ and "SLURM_NODEID" in os.environ


def get_nn_from_ddp_module(model: nn.Module) -> nn.Module:
    """
    Return a real model from a torch.nn.DataParallel,
    torch.nn.parallel.DistributedDataParallel, or
    apex.parallel.DistributedDataParallel.

    Args:
        model: A model, or DataParallel wrapper.

    Returns:
        A model
    """
    if check_ddp_wrapped(model):
        model = model.module
    return model


def get_rank() -> int:
    """
    Returns the rank of the current worker.

    Returns:
        int: ``rank`` if torch.distributed is initialized, otherwise ``0``
    """
    if check_torch_distributed_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_distributed_mean(value: Union[float, torch.Tensor]):
    """Computes distributed mean among all nodes."""
    if check_torch_distributed_initialized():
        # Fix for runtime warning:
        # To copy construct from a tensor, it is recommended to use
        # sourceTensor.clone().detach() or
        # sourceTensor.clone().detach().requires_grad_(True),
        # rather than torch.tensor(sourceTensor).
        if torch.is_tensor(value):
            value = value.clone().detach().to(device=f"cuda:{torch.cuda.current_device()}")
        else:
            value = torch.tensor(
                value,
                dtype=torch.float,
                device=f"cuda:{torch.cuda.current_device()}",
                requires_grad=False,
            )
        torch.distributed.all_reduce(value)
        value = float(value.item() / torch.distributed.get_world_size())
    return value


__all__ = [
    "check_ddp_wrapped",
    "check_amp_available",
    "check_torch_distributed_initialized",
    "check_slurm_available",
    "maybe_torch_distributed_barrier",
    "get_nn_from_ddp_module",
    "get_rank",
    "get_distributed_mean",
]
