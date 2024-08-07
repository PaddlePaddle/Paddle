# copyright (c) 2022 paddlepaddle authors. all rights reserved.
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

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

from paddle.nn import Linear
from paddle.quantization.base_quanter import BaseQuanter
from paddle.quantization.factory import quanter

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

    import paddle

linear_quant_axis = 1


@quanter("CustomizedQuanter")
class CustomizedQuanterLayer(BaseQuanter):
    def __init__(self, layer, bit_length=8, kwargs1=None):
        super().__init__()
        self._layer = layer
        self._bit_length = bit_length
        self._kwargs1 = kwargs1

    def scales(self) -> paddle.Tensor | np.ndarray:
        return None

    def bit_length(self):
        return self._bit_length

    def quant_axis(self) -> int | Iterable:
        return linear_quant_axis if isinstance(self._layer, Linear) else None

    def zero_points(self) -> paddle.Tensor | np.ndarray:
        return None

    def forward(self, input):
        return input


class TestCustomizedQuanter(unittest.TestCase):
    def test_details(self):
        layer = Linear(5, 5)
        bit_length = 4
        quanter = CustomizedQuanter(  # noqa: F821
            bit_length=bit_length, kwargs1="test"
        )
        quanter = quanter._instance(layer)
        self.assertEqual(quanter.bit_length(), bit_length)
        self.assertEqual(quanter.quant_axis(), linear_quant_axis)
        self.assertEqual(quanter._kwargs1, 'test')


if __name__ == '__main__':
    unittest.main()
