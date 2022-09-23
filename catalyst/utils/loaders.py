import warnings
from copy import copy
from typing import Dict, Union

from catalyst.utils.distributed import get_rank
from torch.utils.data import DataLoader, DistributedSampler


def get_native_batch_from_loader(loader: DataLoader, batch_index: int = 0):
    """
    Returns a batch from experiment loader

    Args:
        loader: Loader to get batch from
        batch_index: Index of batch to take from dataset of the loader

    Returns:
        batch from loader
    """
    dataset = loader.dataset
    collate_fn = loader.collate_fn
    return collate_fn([dataset[batch_index]])


def get_native_batch_from_loaders(
    loaders: Dict[str, DataLoader], loader: Union[str, int] = 0, batch_index: int = 0,
):
    """
    Returns a batch from experiment loaders by its index or name.

    Args:
        loaders (Dict[str, DataLoader]): Loaders list to get loader from
        loader (Union[str, int]): Loader name or its index, default is zero
        batch_index: Index of batch to take from dataset of the loader

    Returns:
        batch from loader

    Raises:
        TypeError: if loader parameter is not a string or an integer
    """
    if isinstance(loader, str):
        loader_instance = loaders[loader]
    elif isinstance(loader, int):
        loader_instance = list(loaders.values())[loader]
    else:
        raise TypeError("Loader parameter must be a string or an integer")

    output = get_native_batch_from_loader(loader=loader_instance, batch_index=batch_index)

    return output


def _force_make_distributed_loader(loader: DataLoader) -> DataLoader:
    """
    Transfers loader to distributed mode. Experimental feature.

    Args:
        loader: pytorch dataloder

    Returns:
        DataLoader: pytorch dataloder with distributed sampler.
    """
    from catalyst.data.sampler import DistributedSamplerWrapper

    sampler = (
        DistributedSampler(dataset=loader.dataset)
        if getattr(loader, "sampler", None) is not None
        else DistributedSamplerWrapper(sampler=loader.sampler)
    )
    loader = DataLoader(
        dataset=copy(loader.dataset),
        batch_size=loader.batch_size,
        # shuffle=loader.shuffle,
        sampler=sampler,
        # batch_sampler=loader.batch_sampler,
        num_workers=loader.num_workers,
        # collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )
    return loader


def validate_loaders(loaders: Dict[str, DataLoader]) -> Dict[str, DataLoader]:
    """
    Check pytorch dataloaders for distributed setup.
    Transfers them to distirbuted mode if necessary.
    (Experimental feature)

    Args:
        loaders (Dict[str, DataLoader]): dictionery with pytorch dataloaders

    Returns:
        Dict[str, DataLoader]: dictionery
            with pytorch dataloaders (with distributed samplers if necessary)
    """
    from catalyst.data.sampler import DistributedSamplerWrapper

    rank = get_rank()
    if rank >= 0:
        for key, value in loaders.items():
            if not isinstance(value.sampler, (DistributedSampler, DistributedSamplerWrapper)):
                warnings.warn(
                    "With distributed training setup, "
                    "you need ``DistributedSampler`` for your ``DataLoader``."
                )
                # loaders[key] = _force_make_distributed_loader(value)
    return loaders





__all__ = [
    "get_native_batch_from_loader",
    "get_native_batch_from_loaders",
    "validate_loaders",
]
