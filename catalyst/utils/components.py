import copy
from typing import Dict, Tuple

import torch
import torch.distributed
from torch import nn

from catalyst.typing import Criterion, Device, Model, Optimizer
from catalyst.utils.distributed import (
    check_amp_available,
    check_ddp_wrapped,
    get_distributed_params,
    get_rank,
)
from catalyst.utils.misc import maybe_recursive_call
from catalyst.utils.torch import get_device


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
    elif isinstance(device, str):
        device = torch.device(device)

    is_amp_enabled = distributed_params.get("amp", False) and check_amp_available()
    model: Model = maybe_recursive_call(model, "to", device=device)

    if check_ddp_wrapped(model):
        pass
    elif get_rank() >= 0:
        # distributed data parallel run (ddp) (with apex support)
        assert isinstance(
            model, nn.Module
        ), "Distributed training is not available for KV model"

        local_rank = distributed_params.pop("local_rank", 0) or 0
        device = f"cuda:{local_rank}"
        model = maybe_recursive_call(model, "to", device=device)

        syncbn = distributed_params.pop("syncbn", False)

        if syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        find_unused = distributed_params.get("find_unused_parameters", False)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )
    else:
        # data parallel run (dp) (with apex support)
        if (
            torch.cuda.device_count() > 1
            and device.type != "cpu"
            and device.index is None
        ):
            if isinstance(model, nn.Module):
                model = nn.DataParallel(model)
            elif isinstance(model, dict):
                model = {k: nn.DataParallel(v) for k, v in model.items()}
            else:
                raise NotImplementedError()

    model: Model = maybe_recursive_call(model, "to", device=device)

    return model, criterion, optimizer, device


__all__ = ["process_components"]
