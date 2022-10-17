import math

import torch
from catalyst.core import CallbackOrder, CallbackNode
from catalyst.utils import get_dictkey_auto_fn
from torch import nn

from catalyst.core import Callback
from pytorch_toolbelt.utils.distributed import all_gather


class PerplexityMetricCallback(Callback):
    """
    Perplexity is a very popular metric in NLP
    especially in Language Modeling task.
    It is 2^cross_entropy.
    """

    def __init__(
        self,
        targets_key: str = "targets",
        predictions_keys: str = "logits",
        prefix: str = "perplexity",
        ignore_index: int = -100,
        base: float = math.e,
    ):
        """
        Args:
            targets_key: input key to use for perplexity calculation,
                target tokens
            predictions_keys: output key to use for perplexity calculation,
                logits of the predicted tokens
            ignore_index: index to ignore, usually pad_index
        """
        super().__init__(
            order=CallbackOrder.metric,
            node=CallbackNode.all,
        )

        self.targets_key = targets_key
        self.output_key = predictions_keys

        self.prefix = prefix
        self.base = base
        self._get_targets = get_dictkey_auto_fn(targets_key)
        self._get_predictions = get_dictkey_auto_fn(predictions_keys)

        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
        self.ignore_index = ignore_index
        self.cross_entropy_samples = None
        self.cross_entropy_sum = None

    def on_loader_start(self, runner: "IRunner"):
        self.cross_entropy_samples = float(0)
        self.cross_entropy_loss = float(0)

    @torch.no_grad()
    def on_batch_end(self, runner: "IRunner") -> None:
        predictions = self._get_predictions(runner.output, self.output_key)
        targets = self._get_targets(runner.input, self.targets_key)
        cross_entropy = self.cross_entropy_loss(predictions, targets)
        self.cross_entropy_samples += float(cross_entropy.sum())
        self.cross_entropy_loss += float(sum(targets != self.ignore_index))

    def on_loader_end(self, runner: "IRunner"):
        cross_entropy_sum = sum(all_gather(self.cross_entropy_sum))
        cross_entropy_samples = sum(all_gather(self.cross_entropy_samples))
        perplexity = self.base ** (cross_entropy_sum / cross_entropy_samples)
        runner.loader_metrics[self.prefix] = float(perplexity)
