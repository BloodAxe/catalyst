from typing import Dict, Union

from catalyst.core import CallbackNode
from catalyst.core import IRunner, Callback, CallbackOrder
from catalyst.utils import get_tensorboard_logger

__all__ = [
    "HyperParametersCallback",
]


class HyperParametersCallback(Callback):
    """
    Callback that logs hyperparameters for training session and target metric value.
    Useful for evaluation of several runs in Tensorboard.
    """

    def __init__(self, hparam_dict: Dict[str, Union[str, bool, int, float]]):
        if "stage" in hparam_dict:
            raise KeyError("Key 'stage' is reserved")
        for key, value in hparam_dict.items():
            if not isinstance(value, (str, float, int, bool)):
                raise ValueError(f"Value of key {key} must be either str,float,int,bool. Got {value}")

        super().__init__(CallbackOrder.Metric, node=CallbackNode.Master)
        self.hparam_dict = hparam_dict

    def on_stage_end(self, state: IRunner):
        logger = get_tensorboard_logger(state)

        hparam_dict = self.hparam_dict.copy()
        hparam_dict["stage"] = state.stage_name

        logger.add_hparams(
            hparam_dict=self.hparam_dict,
            metric_dict=state.best_valid_metrics,
        )
