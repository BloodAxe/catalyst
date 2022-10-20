from torch.utils.tensorboard import SummaryWriter


def get_tensorboard_logger(runner, tensorboard_callback_name: str = "_tensorboard") -> SummaryWriter:
    tb_callback = runner.callbacks[tensorboard_callback_name]
    if runner.loader_name not in tb_callback.loggers:
        raise RuntimeError(f"Cannot find Tensorboard logger for loader {runner.loader_name}")
    return tb_callback.loggers[runner.loader_name]
