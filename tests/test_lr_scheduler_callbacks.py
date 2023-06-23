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
)
from catalyst.runners import SupervisedRunner


def test_reducelronplateaucallback():
    """Tests EarlyStoppingCallback."""
    early_stop = ReduceLROnPlateauCallback(
        patience=10,
        multiplier=0.5,
        metric_to_monitor="loss",
        minimize=False,  # Intentionally maximize
        min_delta=1e-3,
        warmup_num_steps=100,
        warmup_lr_fraction=0.1,
    )

    model = nn.Sequential(
        collections.OrderedDict([("fc1", nn.Linear(32, 32)), ("fc2", nn.Linear(32, 1)), ("act", nn.ReLU())])
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        [
            {"lr": 1e-4, "params": model.fc1.parameters()},
            {"lr": 1e-2, "params": model.fc2.parameters()},
        ],
        lr=1e-1,
    )

    optimizer_callback = OptimizerCallback()
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
            early_stop,
            TensorboardLogger(),
            OptimizerLoggerCallback(),
        ],
        num_epochs=50,
        logdir="./test_reducelronplateaucallback",
        verbose=True,
    )

    print(optimizer.param_groups[0]["lr"])
    assert math.fabs(optimizer.param_groups[0]["lr"] - 1e-1 * 0.5 * 0.5 * 0.5 * 0.5) < 1e-6


def test_cosinedecayschedulercallback():
    """Tests CosineDecaySchedulerCallback."""

    model = nn.Sequential(
        collections.OrderedDict([("fc1", nn.Linear(32, 32)), ("fc2", nn.Linear(32, 1)), ("act", nn.ReLU())])
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        [
            {"lr": 1e-4, "params": model.fc1.parameters()},
            {"lr": 1e-2, "params": model.fc2.parameters()},
        ],
        lr=1e-1,
    )

    optimizer_callback = OptimizerCallback()
    runner = SupervisedRunner(
        device="cuda",
    )

    inputs = np.random.normal(0, 1, (8192, 32)).astype(np.float32)
    targets = -np.ones((256, 1)).astype(np.float32)
    dataset = list(zip(inputs, targets))
    batch_size = 4
    loaders = collections.OrderedDict(
        [
            ("train", DataLoader(dataset, batch_size=batch_size, shuffle=True)),
            ("valid", DataLoader(dataset, batch_size=batch_size)),
        ]
    )
    num_epochs = 50

    scheduler = CosineDecaySchedulerCallback(
        warmup_num_steps=(len(loaders["train"]) // 2),
        warmup_lr_fraction=0.01,
        final_lr_fraction=0.5,
        num_flat_epochs=10,
        num_cooldown_epochs=15,
    )

    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        callbacks=[
            optimizer_callback,
            scheduler,
            TensorboardLogger(),
            OptimizerLoggerCallback(),
        ],
        num_epochs=num_epochs,
        logdir="./test_cosinedecayschedulercallback",
        verbose=True,
    )

    print(optimizer.param_groups[0]["lr"])
