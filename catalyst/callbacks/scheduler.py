import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner

__all__ = ["ISchedulerCallback", "SchedulerCallback"]


class ISchedulerCallback(Callback):
    """Scheduler callback interface, abstraction over scheduler step."""

    pass
    
    @property
    def has_finite_number_of_epochs(self) -> bool:
        raise NotImplementedError()


class SchedulerCallback(ISchedulerCallback):
    """Callback for wrapping schedulers.

    Notebook API example:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst.dl import (
            SupervisedRunner, AccuracyCallback,
            CriterionCallback, SchedulerCallback,
        )

        num_samples, num_features = 10_000, 10
        n_classes = 10
        X = torch.rand(num_samples, num_features)
        y = torch.randint(0, n_classes, [num_samples])
        loader = DataLoader(TensorDataset(X, y), batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        model = torch.nn.Linear(num_features, n_classes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

        runner = SupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir="./logdir",
            num_epochs=5,
            verbose=False,
            main_metric="accuracy03",
            minimize_metric=False,
            callbacks=[
                AccuracyCallback(
                    accuracy_args=[1, 3, 5]
                ),
                SchedulerCallback(reduced_metric="loss")
            ]
        )

    Config API usage example:

    .. code-block:: yaml

        stages:
          ...
          scheduler_params:
            scheduler: MultiStepLR
            milestones: [1]
            gamma: 0.3
          ...
          stage_N:
            ...
            callbacks_params:
              ...
              scheduler:
                callback: SchedulerCallback
                # arguments for SchedulerCallback
                reduced_metric: loss
          ...
    """

    def __init__(
        self,
        scheduler_key: str = None,
        mode: str = None,
        reduced_metric: str = None,
    ):
        """
        Args:
            scheduler_key: scheduler name, if ``None``,
                default is ``None``.
            mode: scheduler mode, should be one of
                ``"epoch"`` or ``"batch"``, default is ``None``.
                If ``None`` and object is instance of ``BatchScheduler``
                or ``OneCycleLRWithWarmup`` then will be used ``"batch"``
                otherwise - ``"epoch"``.
            reduced_metric: metric name to forward to scheduler
                object, if ``None`` then will be used main metric
                specified in experiment.
        """
        super().__init__(order=CallbackOrder.scheduler, node=CallbackNode.all)
        self.scheduler_key = scheduler_key
        self.mode = mode
        self.reduced_metric = reduced_metric

    @staticmethod
    def _scheduler_step(
        scheduler,
        reduced_metric=None,
    ):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(reduced_metric)
        else:
            scheduler.step()

    def step_batch(self, runner: "IRunner") -> None:
        """Update learning rate and momentum in runner.

        Args:
            runner: current runner
        """
        self._scheduler_step(scheduler=self._scheduler)

    def step_epoch(self, runner: "IRunner") -> None:
        """Update momentum in runner.

        Args:
            runner: current runner
        """
        reduced_metric = runner.valid_metrics[self.reduced_metric]
        self._scheduler_step(scheduler=self._scheduler, reduced_metric=reduced_metric)

    def on_stage_start(self, runner: "IRunner") -> None:
        """Stage start hook.

        Args:
            runner: current runner
        """
        self.reduced_metric = self.reduced_metric or runner.main_metric

        scheduler = runner.get_attr(key="scheduler", inner_key=self.scheduler_key)
        assert scheduler is not None
        self._scheduler = scheduler

    def on_loader_start(self, runner: "IRunner") -> None:
        """Loader start hook.

        Args:
            runner: current runner
        """
        pass

    def on_batch_end(self, runner: "IRunner") -> None:
        """Batch end hook.

        Args:
            runner: current runner
        """
        if runner.is_train_loader and self.mode == "batch":
            self.step_batch(runner=runner)

    def on_epoch_end(self, runner: "IRunner") -> None:
        """Epoch end hook.

        Args:
            runner: current runner
        """
        if self.mode == "epoch":
            self.step_epoch(runner=runner)
