# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import paddle

from ..base_observer import BaseObserver
from ..factory import ObserverFactory


class GroupWiseWeightObserver(ObserverFactory):
    r"""
    It collects channel-wise maximum absolute values of target weights.
    Args:
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
    Examples:
       .. code-block:: python
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import AbsMaxChannelWiseWeightObserver
            quanter = AbsMaxChannelWiseWeightObserver()
            q_config = QuantConfig(activation=None, weight=quanter)
    """

    def __init__(self, quant_bits=8, group_size=128):
        super().__init__(quant_bits=quant_bits)

    def _get_class(self):
        return GroupWiseWeightObserverLayer


class GroupWiseWeightObserverLayer(BaseObserver):
    def __init__(self, layer, quant_bits=8, group_size=128):
        super().__init__()
        self._quant_bits = quant_bits
        self.group_size = group_size
        self._layer = layer
        self._max = None
        self._scale = None
        self._zero_point = None

    def forward(self, inputs):
        self._max = self._cal_abs_max(inputs)
        return inputs

    def _cal_abs_max(self, inputs):
        """Use group_size to group the input, then use the
        absmax method to calculate the scale
        """
        input_shape = inputs.shape
        assert (
            self.group_size == 64 or self.group_size == 128
        ), "group_size only support 64 or 128"
        assert (
            inputs.shape[0] % self.group_size == 0
        ), "group_size must be a factor of input channels"
        assert len(inputs.shape) == 2, "Currently only support 2D tensor"
        input_processed = inputs.transpose([1, 0]).reshape(
            [input_shape[1], input_shape[0] // self.group_size, self.group_size]
        )

        abs_max_values = paddle.max(paddle.abs(input_processed), axis=2).cast(
            "float32"
        )
        abs_max_values = paddle.where(
            abs_max_values == np.float32(0), np.float32(1e-8), abs_max_values
        )
        abs_max_values = abs_max_values.transpose([1, 0])
        return abs_max_values

    def min_value(self) -> float:
        return 0.0

    def max_value(self) -> float:
        return self._max

    def bit_length(self):
        return self._quant_bits

    def quant_axis(self):
        return -1

    def cal_thresholds(self):
        """Compute thresholds for MAX function."""
        if self._scale is None:
            self._scale = self._max
        self._zero_point = paddle.zeros_like(self._scale)

    def scales(self):
        """Return output scales."""
        if self._scale is None:
            self.cal_thresholds()
        return self._scale

    def zero_points(self):
        """Return output zero points."""
        if self._zero_point is None:
            self.cal_thresholds()
        return self._zero_point
