from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Union

from catalyst.core.callback import Callback
from catalyst.core.runner import IStageBasedRunner
from catalyst.experiments.experiment import Experiment
from catalyst.typing import Criterion, Model, Optimizer, Scheduler
from torch.utils.data import DataLoader, Dataset


class Runner(IStageBasedRunner):
    """
    Deep Learning Runner for supervised, unsupervised, gan, etc runs.
    """

    _experiment_fn: Callable = Experiment

    def _init(self, **kwargs):
        self.experiment: Experiment = None

    def train(
        self,
        *,
        model: Model,
        criterion: Union[Criterion, Mapping[str, Criterion]] = None,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
        datasets: "OrderedDict[str, Union[Dataset, Dict, Any]]" = None,
        loaders: "OrderedDict[str, DataLoader]" = None,
        callbacks: "Union[List[Callback], OrderedDict[str, Callback]]" = None,
        logdir: str = None,
        num_epochs: int = 1,
        valid_loader: str = "valid",
        main_metric: str = "loss",
        minimize_metric: bool = True,
        verbose: bool = False,
        stage_kwargs: Dict = None,
        checkpoint_data: Dict = None,
        fp16: Union[Dict, bool] = None,
        timeit: bool = False,
        initial_seed: int = 42,
        state_kwargs: Dict = None,
    ) -> None:
        """
        Starts the train stage of the model.

        Args:
            model: model to train
            criterion: criterion function for training
            optimizer: optimizer for training
            scheduler: scheduler for training
            datasets (OrderedDict[str, Union[Dataset, Dict, Any]]): dictionary
                with one or several  ``torch.utils.data.Dataset``
                for training, validation or inference
                used for Loaders automatic creation
                preferred way for distributed training setup
            loaders (OrderedDict[str, DataLoader]): dictionary
                with one or several ``torch.utils.data.DataLoader``
                for training, validation or inference
            callbacks (Union[List[Callback], OrderedDict[str, Callback]]):
                list or dictionary with Catalyst callbacks
            logdir: path to output directory
            resume: path to checkpoint for model
            num_epochs: number of training epochs
            valid_loader: loader name used to calculate
                the metrics and save the checkpoints. For example,
                you can pass `train` and then
                the metrics will be taken from `train` loader.
            main_metric: the key to the name of the metric
                by which the checkpoints will be selected.
            minimize_metric: flag to indicate whether
                the ``main_metric`` should be minimized.
            verbose: if `True`, it displays the status of the training
                to the console.
            stage_kwargs: additional params for stage
            checkpoint_data: additional data to save in checkpoint,
                for example: ``class_names``, ``date_of_training``, etc
            fp16 (Union[Dict, bool]): If not None, then sets training to FP16.
                See https://nvidia.github.io/apex/amp.html#properties
                if fp16=True, params by default will be ``{"opt_level": "O1"}``
            distributed: if `True` will start training
                in distributed mode.
                Note: Works only with python scripts. No jupyter support.
            check: if True, then only checks that pipeline is working
                (3 epochs only with 3 batches per loader)
            overfit: if True, then takes only one batch per loader
                for model overfitting, for advance usage please check
                ``BatchOverfitCallback``
            timeit: if True, computes the execution time
                of training process and displays it to the console.
            initial_seed: experiment's initial seed value
            state_kwargs: deprecated, use `stage_kwargs` instead

        Raises:
            NotImplementedError: if both `resume` and `CheckpointCallback`
                already exist
        """
        assert state_kwargs is None or stage_kwargs is None

        experiment = self._experiment_fn(
            stage="train",
            model=model,
            datasets=datasets,
            loaders=loaders,
            callbacks=callbacks,
            logdir=logdir,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            valid_loader=valid_loader,
            main_metric=main_metric,
            minimize_metric=minimize_metric,
            verbose=verbose,
            check_time=timeit,
            stage_kwargs=stage_kwargs or state_kwargs,
            checkpoint_data=checkpoint_data,
            distributed_params=fp16,
            initial_seed=initial_seed,
        )
        self.experiment = experiment
        self.run_experiment(self.experiment)


__all__ = ["Runner"]
