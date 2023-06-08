import dataclasses
import shutil
from pprint import pprint

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from catalyst.callbacks import TensorboardLogger
from catalyst.callbacks.metrics.global_binary_dice_score import GlobalBinaryDiceScore


@dataclasses.dataclass
class DummyRunner:
    output: dict = dataclasses.field(default_factory=dict)
    input: dict = dataclasses.field(default_factory=dict)
    loader_metrics: dict = dataclasses.field(default_factory=dict)
    callbacks:dict = dataclasses.field(default_factory=dict)
    loader_name: str = "train"
    global_epoch: int = 0

def test_global_binary_dice_score():
    metric = GlobalBinaryDiceScore(
        predictions_key="logits",
        targets_key="targets",
        activation=None,
        threshold=np.linspace(0, 1, num=50, endpoint=True),
        metric_name="metrics/global_dice",
        beta=1.0
    )
    
    shutil.rmtree("test_global_binary_dice_score")
    logger = TensorboardLogger()
    logger.loggers["train"] = SummaryWriter("test_global_binary_dice_score")
    
    targets = torch.from_numpy(np.random.randint(0, 2, (10, 1, 1000, 1000))).float()
    predictions = (1 - targets) * torch.randn_like(targets) * 0.1 + targets * torch.randn_like(targets)
    
    runner = DummyRunner(
        output={"logits": predictions},
        input={"targets": targets},
        callbacks=dict(_tensorboard=logger)
    )
    
    metric.on_loader_start(runner)
    metric.on_batch_end(runner)
    metric.on_loader_end(runner)
    
    pprint(runner.loader_metrics)
    
    
    