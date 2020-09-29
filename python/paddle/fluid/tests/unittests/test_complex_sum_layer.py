# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
import paddle.fluid.dygraph as dg
from numpy.random import random as rand

import paddle.fluid as fluid
from paddle import complex as cpx


class TestComplexSumLayer(unittest.TestCase):
    def setUp(self):
        self._dtype = "float64"
        self._places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            self._places.append(fluid.CUDAPlace(0))

    def test_complex_x(self):
        input = rand([2, 10, 10]).astype(self._dtype) + 1j * rand(
            [2, 10, 10]).astype(self._dtype)
        for place in self._places:
            with dg.guard(place):
                var_x = dg.to_variable(input)
                result = cpx.sum(var_x, dim=[1, 2]).numpy()
                target = np.sum(input, axis=(1, 2))
                self.assertTrue(np.allclose(result, target))


if __name__ == '__main__':
    unittest.main()
