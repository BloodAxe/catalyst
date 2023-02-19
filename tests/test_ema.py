import collections

import numpy as np
import torch.optim
from torch import nn
from torch.utils.data import DataLoader

from catalyst.callbacks import (
    OptimizerCallback,
    TensorboardLogger,
    OptimizerLoggerCallback,
    EMACallback,
)
from catalyst.callbacks.ema import ExpEMADecay, BetaDecay, ThresholdDecay
from catalyst.runners import SupervisedRunner


def test_ema_decay():
    import matplotlib.pyplot as plt

    x = np.linspace(1, 1000000, num=1024)

    plt.figure()
    plt.plot(
        x,
        ExpEMADecay(decay=0.9999, beta=5)(x, total_steps=max(x)),
        label="decay=0.9999, beta=5",
    )
    plt.plot(
        x,
        ExpEMADecay(decay=0.9999, beta=10)(x, total_steps=max(x)),
        label="decay=0.9999, beta=10",
    )
    plt.plot(
        x,
        ExpEMADecay(decay=0.9999, beta=15)(x, total_steps=max(x)),
        label="decay=0.9999, beta=15",
    )
    plt.plot(
        x,
        ExpEMADecay(decay=0.9999, beta=20)(x, total_steps=max(x)),
        label="decay=0.9999, beta=20",
    )
    plt.tight_layout()
    plt.legend()
    plt.title("ExpEMADecay")
    plt.show()


def test_beta_decay():
    import matplotlib.pyplot as plt

    x = np.linspace(1, 1000000, num=1024)

    plt.figure()
    plt.plot(
        x,
        BetaDecay(beta=2)(x, total_steps=max(x)),
        label="beta=2",
    )
    plt.plot(
        x,
        BetaDecay(beta=3)(x, total_steps=max(x)),
        label="beta=3",
    )
    plt.plot(
        x,
        BetaDecay(beta=4)(x, total_steps=max(x)),
        label="beta=4",
    )
    plt.plot(
        x,
        BetaDecay(beta=5)(x, total_steps=max(x)),
        label="beta=5",
    )
    plt.tight_layout()
    plt.legend()
    plt.title("BetaDecay")
    plt.show()


def test_ema_disabled():
    """Tests EarlyStoppingCallback."""

    model = nn.Sequential(
        collections.OrderedDict(
            [
                ("fc1", nn.Linear(32, 32)),
                ("bn1", nn.BatchNorm1d(32)),
                ("act1", nn.ReLU()),
                ("fc2", nn.Linear(32, 1)),
            ]
        )
    ).cuda()
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
    targets = -np.ones((8192, 1)).astype(np.float32)
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
            TensorboardLogger(),
            OptimizerLoggerCallback(),
        ],
        num_epochs=50,
        logdir="./test_ema_disabled",
        verbose=True,
    )


def test_ema():
    """Tests EarlyStoppingCallback."""
    ema = EMACallback(decay=ThresholdDecay(decay=0.999))

    model = nn.Sequential(
        collections.OrderedDict(
            [
                ("fc1", nn.Linear(32, 32)),
                ("bn1", nn.BatchNorm1d(32)),
                ("act1", nn.ReLU()),
                ("fc2", nn.Linear(32, 1)),
            ]
        )
    ).cuda()
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
    targets = -np.ones((8192, 1)).astype(np.float32)
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


def test_plot_ema_schedules():
    import matplotlib.pyplot as plt

    total_steps = 1000000
    steps = np.linspace(1, total_steps, total_steps, endpoint=True)

    plt.figure()

    for name, ema_algs in [
        ("beta", BetaDecay(beta=15)),
        ("threshold", ThresholdDecay(0.9998)),
        ("exp", ExpEMADecay(decay=0.9998, beta=4)),
    ]:
        plt.plot(steps, ema_algs(steps, total_steps), label=name)

    plt.legend()
    plt.tight_layout()
    plt.show()
