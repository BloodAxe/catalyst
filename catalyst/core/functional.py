import collections
from collections import OrderedDict
from typing import Dict, List, Union

from catalyst.core.callback import Callback, CallbackNode, CallbackWrapper
from catalyst.utils.distributed import get_rank


def get_original_callback(callback: Callback) -> Callback:
    """Get original callback (if it has wrapper)

    Args:
        callback: callback to unpack

    Returns:
        callback inside wrapper
    """
    while isinstance(callback, CallbackWrapper):
        callback = callback.callback
    return callback


def check_callback_isinstance(callback: Callback, class_or_tuple) -> bool:
    """Check if callback is the same type as required ``class_or_tuple``

    Args:
        callback: callback to check
        class_or_tuple: class_or_tuple to compare with

    Returns:
        bool: true if first object has the required type
    """
    callback = get_original_callback(callback)
    return isinstance(callback, class_or_tuple)


def sort_callbacks_by_order(callbacks: Union[List, Dict, OrderedDict]) -> OrderedDict:
    """Creates an sequence of callbacks and sort them.

    Args:
        callbacks: either list of callbacks or ordered dict

    Returns:
        sequence of callbacks sorted by ``callback order``

    Raises:
        TypeError: if `callbacks` is out of
            `None`, `dict`, `OrderedDict`, `list`
    """
    if callbacks is None:
        output = OrderedDict()
    elif isinstance(callbacks, (dict, OrderedDict)):
        output = [(k, v) for k, v in callbacks.items()]
        output = sorted(output, key=lambda x: x[1].order)
        output = OrderedDict(output)
    elif isinstance(callbacks, list):
        output = sorted(callbacks, key=lambda x: x.order)
        output = OrderedDict([(i, value) for i, value in enumerate(output)])
    else:
        raise TypeError(f"Callbacks must be either Dict/OrderedDict or list, " f"got {type(callbacks)}")

    return output


def filter_callbacks_by_node(callbacks: Union[Dict, OrderedDict]) -> collections.OrderedDict:
    """
    Filters callbacks based on running node.
    Deletes worker-only callbacks from ``CallbackNode.Master``
    and master-only callbacks from ``CallbackNode.Worker``.

    Args:
        callbacks (Union[Dict, OrderedDict]): callbacks

    Returns:
        OrderedDict: filtered callbacks dictionary.
    """
    rank = get_rank()
    if rank == 0:  # master node or single-gpu mode
        output = [
            (key, callback)
            for key, callback in callbacks.items()
            if callback.node in {CallbackNode.all, CallbackNode.master}
        ]
    else:  # worker node
        output = [
            (key, callback)
            for key, callback in callbacks.items()
            if callback.node in {CallbackNode.all, CallbackNode.worker}
        ]
    return collections.OrderedDict(output)


__all__ = [
    "sort_callbacks_by_order",
    "filter_callbacks_by_node",
    "get_original_callback",
    "check_callback_isinstance",
]
