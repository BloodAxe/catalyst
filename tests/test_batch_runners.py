import torch
from torch import nn

from catalyst.batch_runners import SlidingWindowBatchRunner

class ReturnOnes(nn.Module):
    def forward(self, x):
        return torch.ones_like(x)

def test_sliding_window_runner():
    runner_1d = SlidingWindowBatchRunner(
        None, None, tile_size=(37,), step_size=(12,), padding_mode="constant", padding_value=0
    )
    model = ReturnOnes()
    inputs = torch.randn(1, 3, 55)
    outputs = runner_1d(model, inputs)
    assert outputs.shape == inputs.shape
    assert torch.all(outputs == 1)

    runner_2d = SlidingWindowBatchRunner(
        None, None, tile_size=(128, 99), step_size=(37, 67), padding_mode="constant", padding_value=0
    )

    inputs = torch.randn(1, 3, 555, 777)
    outputs = runner_2d(model, inputs)
    assert outputs.shape == inputs.shape
    assert torch.all(outputs == 1)

    runner_3d = SlidingWindowBatchRunner(
        None, None, tile_size=(128, 99, 64), step_size=(37, 67, 33), padding_mode="constant", padding_value=0
    )

    inputs = torch.randn(1, 3, 555, 777, 888)
    outputs = runner_3d(model, inputs)
    assert outputs.shape == inputs.shape
    assert torch.all(outputs == 1)
