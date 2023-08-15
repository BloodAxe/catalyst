import itertools
from typing import Tuple, List

import torch

from pytorch_toolbelt.inference.functional import unpad_xyxy_bboxes, pad_image_tensor, unpad_image_tensor
from torch import nn


class BasicBatchRunner:
    def __init__(self, input_keys, output_keys):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def get_inputs(self, batch):
        return batch[self.input_keys]

    def get_outputs(self, batch):
        return batch[self.output_keys]

    def __call__(self, model, batch):
        inputs = self.get_inputs(batch)
        outputs = model(inputs)
        return self.get_outputs(outputs)


class SlidingWindowBatchRunner:
    """
    Sliding inference batch runner that process the input using a sliding window approach.
    Use cases:
    1) Processing input as is impossible because of memory restrictions (Inputs and outputs must fit for obvious reasons)
    2) You want to validate the model on inputs of original resolution
    3) You trained transformer model on fixed resolution and want to run validation on images of different resolutions

    """

    def __init__(self, input_keys, output_keys, tile_size, step_size, padding_mode="constant", padding_value=0):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.tile_size = tile_size
        self.step_size = step_size
        self.num_spatial_dims = len(tile_size)
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def get_inputs(self, batch):
        if self.input_keys is None:
            return batch
        else:
            return batch[self.input_keys]

    def get_outputs(self, outputs):
        if self.output_keys is None:
            return outputs
        else:
            return {self.output_keys: outputs}

    def __call__(self, model, batch):
        inputs = self.get_inputs(batch)

        spatial_dims = inputs.shape[-self.num_spatial_dims :]
        if len(inputs.size()) != len(spatial_dims) + 2:
            raise ValueError(f"Expected {len(spatial_dims) + 2} spatial dimensions, got {len(inputs.size())}")

        spatial_coordinates = [
            compute_tile_coordinates(tile_size, tile_step, length)
            for tile_size, tile_step, length in zip(self.tile_size, self.step_size, spatial_dims)
        ]
        coordinates = itertools.product(*spatial_coordinates)

        full_res_output = None
        full_res_accumulator = None
        for c in coordinates:
            roi = slice(0, None), slice(0, None), *c
            tile = inputs[roi]
            tile, pad = pad_image_tensor(tile, self.tile_size)
            outputs = model(tile)
            outputs = unpad_image_tensor(outputs, pad)

            if full_res_output is None:
                batch_size = outputs.shape[0]
                channels = outputs.shape[1]
                full_res_output = torch.zeros(
                    [batch_size, channels, *spatial_dims], dtype=torch.float32, device=outputs.device
                )
                full_res_accumulator = torch.zeros(
                    [batch_size, 1, *spatial_dims], dtype=torch.float32, device=outputs.device
                )
                full_res_output[roi] += outputs
                full_res_accumulator[roi] += 1

        full_res_output /= full_res_accumulator

        return self.get_outputs(full_res_output)


def compute_tile_coordinates(tile_size: int, step_size: int, length: int) -> List[Tuple[int, int]]:
    """
    Computes the coordinates of the tiles in a sliding window.

    Parameters
    ----------
    tile_size : int
        The size of the tiles.
    step_size : int
        The step size of the sliding window.
    length : int
        The length of the sequence.

    Returns
    -------
    tile_coordinates : list
        The coordinates of the tiles in the sliding window.
    """
    tile_coordinates = []
    if tile_size < length:
        return [slice(0, length)]

    for i in range(0, length - tile_size + 1, step_size):
        tile_coordinates.append(slice(i, i + tile_size))
    return tile_coordinates


if __name__ == "__main__":
    runner_1d = SlidingWindowBatchRunner(
        None, None, tile_size=(128,), step_size=(37,), padding_mode="constant", padding_value=0
    )
    model = nn.Identity()
    inputs = torch.randn(1, 3, 555)
    outputs = runner_1d(model, inputs)
    assert outputs.shape == inputs.shape

    runner_2d = SlidingWindowBatchRunner(
        None, None, tile_size=(128, 99), step_size=(37, 67), padding_mode="constant", padding_value=0
    )
    model = nn.Identity()
    inputs = torch.randn(1, 3, 555, 777)
    outputs = runner_2d(model, inputs)
    assert outputs.shape == inputs.shape

    runner_3d = SlidingWindowBatchRunner(
        None, None, tile_size=(128, 99, 64), step_size=(37, 67, 33), padding_mode="constant", padding_value=0
    )
    model = nn.Identity()
    inputs = torch.randn(1, 3, 555, 777, 888)
    outputs = runner_3d(model, inputs)
    assert outputs.shape == inputs.shape
