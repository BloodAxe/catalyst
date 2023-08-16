import itertools
from typing import Tuple, List

import torch
from pytorch_toolbelt.inference.functional import (
    pad_tensor_to_size,
)


class SlidingWindowBatchRunner:
    """
    Sliding inference batch runner that process the input using a sliding window approach.
    Use cases:
    1) Processing input as is impossible because of memory restrictions (Inputs and outputs must fit for obvious reasons)
    2) You want to validate the model on inputs of original resolution
    3) You trained transformer model on fixed resolution and want to run validation on images of different resolutions

    """

    def __init__(
        self,
        input_keys,
        output_keys,
        tile_size: Tuple[int, ...],
        step_size: Tuple[int, ...],
        padding_mode="constant",
        padding_value=0,
    ):
        if len(tile_size) != len(step_size):
            raise ValueError("Tile size and step size must have the same length")
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

        spatial_sizes = inputs.shape[-self.num_spatial_dims :]
        if len(inputs.size()) != len(spatial_sizes) + 2:
            raise ValueError(f"Expected {len(spatial_sizes) + 2} spatial dimensions, got {len(inputs.size())}")

        spatial_coordinates = [
            self.compute_tile_coordinates(tile_size, tile_step, length)
            for tile_size, tile_step, length in zip(self.tile_size, self.step_size, spatial_sizes)
        ]
        coordinates = itertools.product(*spatial_coordinates)

        full_res_output = None
        full_res_accumulator = None
        for c in coordinates:
            roi = slice(None), slice(None), *c
            tile = inputs[roi]
            tile, unpad_crop = pad_tensor_to_size(tile, self.tile_size)
            outputs = model(tile)
            outputs = outputs[unpad_crop]

            if full_res_output is None:
                batch_size = outputs.shape[0]
                channels = outputs.shape[1]
                full_res_output = torch.zeros(
                    [batch_size, channels, *spatial_sizes], dtype=torch.float32, device=outputs.device
                )
                full_res_accumulator = torch.zeros(
                    [batch_size, 1, *spatial_sizes], dtype=torch.float32, device=outputs.device
                )

            full_res_output[roi] += outputs
            full_res_accumulator[roi] += 1

        full_res_output /= full_res_accumulator

        return self.get_outputs(full_res_output)

    @staticmethod
    def compute_tile_coordinates(tile_size: int, step_size: int, length: int) -> List[slice]:
        """
        Computes the coordinates of the tiles in a sliding window.

        :param tile_size: The window size.
        :param step_size: The step size of the sliding window.
        :param length: The length of the sequence.
        :return: List of slices
        """
        if step_size > tile_size:
            raise ValueError("Step size must be smaller than tile size")

        if length <= step_size:
            return [slice(0, length)]

        num_tiles = length // step_size + 1
        tile_coordinates = []
        for tile in range(num_tiles):
            start = tile * step_size
            end = start + tile_size
            if end > length:
                start = length - tile_size
                end = length
                tile_coordinates.append(slice(start, end))
                break
            else:
                tile_coordinates.append(slice(start, end))

        return tile_coordinates
