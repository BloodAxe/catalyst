import inspect
import shutil
from datetime import datetime
from itertools import tee
from pathlib import Path
from typing import Any, Callable, List, Union, Iterable, Optional

import numpy as np


def pairwise(iterable: Iterable[Any]) -> Iterable[Any]:
    """Iterate sequences by pairs.

    Examples:
        >>> for i in pairwise([1, 2, 5, -3]):
        >>>     print(i)
        (1, 2)
        (2, 5)
        (5, -3)

    Args:
        iterable: Any iterable sequence

    Returns:
        pairwise iterator
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def make_tuple(tuple_like):
    """Creates a tuple if given ``tuple_like`` value isn't list or tuple.

    Args:
        tuple_like: tuple like object - list or tuple

    Returns:
        tuple or list
    """
    tuple_like = tuple_like if isinstance(tuple_like, (list, tuple)) else (tuple_like, tuple_like)
    return tuple_like


def args_are_not_none(*args: Optional[Any]) -> bool:
    """Check that all arguments are not ``None``.

    Args:
        *args: values  # noqa: RST213

    Returns:
         bool: True if all value were not None, False otherwise
    """
    if args is None:
        return False

    for arg in args:
        if arg is None:
            return False

    return True


def find_value_ids(it: Iterable[Any], value: Any) -> List[int]:
    """
    Args:
        it: list of any
        value: query element

    Returns:
        indices of the all elements equal x0
    """
    if isinstance(it, np.ndarray):
        inds = list(np.where(it == value)[0])
    else:  # could be very slow
        inds = [i for i, el in enumerate(it) if el == value]
    return inds


def maybe_recursive_call(
    object_or_dict,
    method: Union[str, Callable],
    recursive_args=None,
    recursive_kwargs=None,
    **kwargs,
):
    """Calls the ``method`` recursively for the ``object_or_dict``.

    Args:
        object_or_dict: some object or a dictionary of objects
        method: method name to call
        recursive_args: list of arguments to pass to the ``method``
        recursive_kwargs: list of key-arguments to pass to the ``method``
        **kwargs: Arbitrary keyword arguments

    Returns:
        result of `method` call
    """
    if isinstance(object_or_dict, dict):
        result = type(object_or_dict)()
        for k, v in object_or_dict.items():
            r_args = None if recursive_args is None else recursive_args[k]
            r_kwargs = None if recursive_kwargs is None else recursive_kwargs[k]
            result[k] = maybe_recursive_call(
                v,
                method,
                recursive_args=r_args,
                recursive_kwargs=r_kwargs,
                **kwargs,
            )
        return result

    r_args = recursive_args or []
    if not isinstance(r_args, (list, tuple)):
        r_args = [r_args]
    r_kwargs = recursive_kwargs or {}
    if isinstance(method, str):
        return getattr(object_or_dict, method)(*r_args, **r_kwargs, **kwargs)
    else:
        return method(object_or_dict, *r_args, **r_kwargs, **kwargs)


def is_exception(ex: Any) -> bool:
    """Check if the argument is of ``Exception`` type."""
    result = (ex is not None) and isinstance(ex, BaseException)
    return result


def copy_directory(input_dir: Path, output_dir: Path) -> None:
    """Recursively copies the input directory.

    Args:
        input_dir: input directory
        output_dir: output directory
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    for path in input_dir.iterdir():
        if path.is_dir():
            path_name = path.name
            copy_directory(path, output_dir / path_name)
        else:
            shutil.copy2(path, output_dir)


def get_utcnow_time(format: str = None) -> str:
    """Return string with current utc time in chosen format.

    Args:
        format: format string. if None "%y%m%d.%H%M%S" will be used.

    Returns:
        str: formatted utc time string
    """
    if format is None:
        format = "%y%m%d.%H%M%S"
    result = datetime.utcnow().strftime(format)
    return result


def format_metric(name: str, value: float) -> str:
    """Format metric.

    Metric will be returned in the scientific format if 4
    decimal chars are not enough (metric value lower than 1e-4).

    Args:
        name: metric name
        value: value of metric

    Returns:
        str: formatted metric
    """
    if value < 1e-4:
        return f"{name}={value:1.3e}"
    return f"{name}={value:.4f}"


def get_fn_default_params(fn: Callable[..., Any], exclude: List[str] = None):
    """Return default parameters of Callable.

    Args:
        fn (Callable[..., Any]): target Callable
        exclude: exclude list of parameters

    Returns:
        dict: contains default parameters of `fn`
    """
    argspec = inspect.getfullargspec(fn)
    default_params = zip(argspec.args[-len(argspec.defaults) :], argspec.defaults)
    if exclude is not None:
        default_params = filter(lambda x: x[0] not in exclude, default_params)
    default_params = dict(default_params)
    return default_params


def get_fn_argsnames(fn: Callable[..., Any], exclude: List[str] = None):
    """Return parameter names of Callable.

    Args:
        fn (Callable[..., Any]): target Callable
        exclude: exclude list of parameters

    Returns:
        list: contains parameter names of `fn`
    """
    argspec = inspect.getfullargspec(fn)
    params = argspec.args + argspec.kwonlyargs
    if exclude is not None:
        params = list(filter(lambda x: x not in exclude, params))
    return params


def fn_ends_with_pass(fn: Callable[..., Any]):
    """
    Check that function end with pass statement
    (probably does nothing in any way).
    Mainly used to filter callbacks with empty on_{event} methods.

    Args:
        fn (Callable[..., Any]): target Callable

    Returns:
        bool: True if there is pass in the first indentation level of fn
        and nothing happens before it, False in any other case.
    """
    source_lines = inspect.getsourcelines(fn)[0]
    if source_lines[-1].strip() == "pass":
        return True
    return False


__all__ = [
    "copy_directory",
    "format_metric",
    "get_fn_default_params",
    "get_fn_argsnames",
    "get_utcnow_time",
    "is_exception",
    "maybe_recursive_call",
    "fn_ends_with_pass",
    "args_are_not_none",
    "make_tuple",
    "pairwise",
    "find_value_ids",
]
