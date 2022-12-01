# flake8: noqa
"""
All utils are gathered in :py:mod:`catalyst.utils` for easier access.
"""


from catalyst.utils.checkpoint import (
    load_checkpoint,
    pack_checkpoint,
    save_checkpoint,
    unpack_checkpoint,
)
from catalyst.utils.components import process_components
from catalyst.utils.dict import (
    get_key_str,
    get_key_none,
    get_key_list,
    get_key_dict,
    get_key_all,
    get_dictkey_auto_fn,
    merge_dicts,
    flatten_dict,
    split_dict_to_subdicts,
)
from catalyst.utils.distributed import (
    get_nn_from_ddp_module,
    get_rank,
    get_distributed_mean,
    check_ddp_wrapped,
    check_torch_distributed_initialized,
    check_slurm_available,
    check_amp_available,
)
from catalyst.utils.loaders import (
    validate_loaders,
    get_native_batch_from_loader,
    get_native_batch_from_loaders,
)
from catalyst.utils.misc import (
    copy_directory,
    format_metric,
    get_fn_default_params,
    get_fn_argsnames,
    get_utcnow_time,
    is_exception,
    maybe_recursive_call,
    fn_ends_with_pass,
)
from catalyst.utils.misc import find_value_ids

from catalyst.utils.seed import set_global_seed
from catalyst.utils.swa import (
    average_weights,
    get_averaged_weights_by_path_mask,
)
from catalyst.utils.sys import (
    get_environment_vars,
    list_conda_packages,
    list_pip_packages,
    dump_environment,
)
from catalyst.utils.torch import (
    any2device,
    get_activation_fn,
    get_available_gpus,
    get_device,
    get_optimizable_params,
    get_param_group_params,
    OptimizerParamGroupParams,
    prepare_cudnn,
    process_model_params,
    get_requires_grad,
    set_requires_grad,
    get_network_output,
    detach,
    trim_tensors,
)
from catalyst.utils.loggers import get_tensorboard_logger
