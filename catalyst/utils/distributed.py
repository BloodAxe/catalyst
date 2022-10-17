import os
import random
import socket
import subprocess
from collections import OrderedDict
from typing import Union

import torch
import torch.distributed
from packaging.version import parse, Version
from torch import nn

from catalyst.utils.misc import get_fn_default_params
from catalyst.utils.torch import get_available_gpus


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
        int: ``rank`` if torch.distributed is initialized, otherwise ``-1``
    """
    if check_torch_distributed_initialized():
        return torch.distributed.get_rank()
    else:
        return -1


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


def get_distributed_params():
    """Returns distributed params for experiment run.

    Returns:
        dictionary with distributed params
    """
    master_port = str(random.randint(5 * 10**4, 6 * 10**4))
    master_addr = "127.0.0.1"
    cur_node, num_nodes = 0, 1

    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", master_addr)
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", master_port)

    workers_per_node = torch.cuda.device_count()
    start_rank = cur_node * workers_per_node
    world_size = num_nodes * workers_per_node

    local_rank = os.getenv("LOCAL_RANK", None)
    rank = os.getenv("RANK", None)
    local_rank, rank = [v and int(v) for v in [local_rank, rank]]
    world_size = int(os.getenv("WORLD_SIZE", world_size))

    output = OrderedDict(
        local_rank=local_rank,
        start_rank=start_rank,
        rank=rank,
        world_size=world_size,
        master_addr=os.environ["MASTER_ADDR"],
        master_port=os.environ["MASTER_PORT"],
    )

    return output


def get_distributed_env(
    local_rank: int,
    rank: int,
    world_size: int,
    use_cuda_visible_devices: bool = True,
):
    """Returns environment copy with extra distributed settings.

    Args:
        local_rank: worker local rank
        rank: worker global rank
        world_size: worker world size
        use_cuda_visible_devices: boolean flag to use available GPU devices

    Returns:
        updated environment copy
    """
    env = os.environ.copy()
    env["RANK"] = str(rank)
    env["WORLD_SIZE"] = str(world_size)
    env["LOCAL_RANK"] = str(local_rank)
    if use_cuda_visible_devices:
        available_gpus = get_available_gpus()
        env["LOCAL_RANK"] = "0"
        env["CUDA_VISIBLE_DEVICES"] = str(available_gpus[local_rank])
    return env


__all__ = [
    "check_ddp_wrapped",
    "check_amp_available",
    "check_torch_distributed_initialized",
    "check_slurm_available",
    "get_nn_from_ddp_module",
    "get_rank",
    "get_distributed_mean",
    "get_distributed_env",
    "get_distributed_params",
]
