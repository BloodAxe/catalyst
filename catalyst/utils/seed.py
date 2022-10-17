import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Sets random seed into PyTorch, Numpy and Random.

    Args:
        seed: random seed
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)


__all__ = ["set_global_seed"]
