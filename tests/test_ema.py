import collections
import math
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import torch.optim
from torch import nn
from torch.utils.data import DataLoader

from catalyst.callbacks import (
    ReduceLROnPlateauCallback,
    OptimizerCallback,
    TensorboardLogger,
    OptimizerLoggerCallback,
    CosineDecaySchedulerCallback,
    EMACallback,
)
from catalyst.runners import SupervisedRunner


def test_ema():
    """Tests EarlyStoppingCallback."""
    ema = EMACallback(decay=0.9999, beta=15)

    model = nn.Sequential(
        collections.OrderedDict(
            [
                ("fc1", nn.Linear(32, 32)),
                ("act1", nn.ReLU()),
                ("fc2", nn.Linear(32, 1)),
            ]
        )
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        [
            {"lr": 1e-4, "params": model.fc1.parameters()},
            {"lr": 1e-2, "params": model.fc2.parameters()},
        ],
        lr=1e-1,
    )

    optimizer_callback = OptimizerCallback(accumulation_steps=4)
    runner = SupervisedRunner(
        device="cuda",
    )

    inputs = np.random.normal(0, 1, (8192, 32)).astype(np.float32)
    targets = -np.ones((256, 1)).astype(np.float32)
    dataset = list(zip(inputs, targets))

    runner.train(
        model=model,
        criterion=criterion,
        loaders=collections.OrderedDict(
            [
                ("train", DataLoader(dataset, batch_size=4, shuffle=True)),
                ("valid", DataLoader(dataset, batch_size=4)),
            ]
        ),
        optimizer=optimizer,
        callbacks=[
            optimizer_callback,
            ema,
            TensorboardLogger(),
            OptimizerLoggerCallback(),
        ],
        num_epochs=50,
        logdir="./test_ema",
        verbose=True,
    )

