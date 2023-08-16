import itertools
from typing import Tuple, List

import torch

from pytorch_toolbelt.inference.functional import (
    unpad_xyxy_bboxes,
    pad_image_tensor,
    unpad_image_tensor,
    pad_tensor_to_size,
)
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

