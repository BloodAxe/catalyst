import logging
from typing import Dict, Tuple

import torch
import torch.distributed
from torch import nn

from catalyst.typing import Criterion, Device, Model, Optimizer
from catalyst.utils.distributed import (
    check_ddp_wrapped,
    get_rank,
    check_torch_distributed_initialized,
)
from catalyst.utils.misc import maybe_recursive_call
from catalyst.utils.torch import get_device

logger = logging.getLogger("catalyst.process_components")


def process_components(
    model: Model,
    criterion: Criterion = None,
    optimizer: Optimizer = None,
    distributed_params: Dict = None,
    device: Device = None,
) -> Tuple[Model, Criterion, Optimizer, Device]:
    """
    Returns the processed model, criterion, optimizer and device.

    Args:
        model: torch model
        criterion: criterion function
        optimizer: optimizer
        distributed_params (dict, optional): dict with the parameters
            for distributed and FP16 method
        device (Device, optional): device

    Returns:
        tuple with processed model, criterion, optimizer and device.

    Raises:
        ValueError: if device is None and TPU available,
            for using TPU need to manualy move model/optimizer
            to a TPU device and pass device to a function.
        NotImplementedError: if model is not nn.Module or dict for multi-gpu,
            nn.ModuleDict for DataParallel not implemented yet
    """
    distributed_params = distributed_params or {}

    if device is None:
        device = get_device()
        logger.info(f"Target device is not specified. Using default device {device}")
    elif isinstance(device, str):
        device = torch.device(device)

    if check_ddp_wrapped(model):
        logger.info(f"Model is already wrapped with DDP. Skipping step")
    elif check_torch_distributed_initialized():
        if not isinstance(model, nn.Module):
            raise ValueError("Distributed training is not available for KV model")

        model = maybe_recursive_call(model, "to", device=device)
        syncbn = distributed_params.get("syncbn", False)

        if syncbn:
            logger.info("Enabling SyncBatchNorm")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        find_unused = distributed_params.get("find_unused_parameters", False)
        local_rank = distributed_params.get("rank", get_rank())

        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )
    else:
        model: Model = maybe_recursive_call(model, "to", device=device)

    return model, criterion, optimizer, device


__all__ = ["process_components"]
