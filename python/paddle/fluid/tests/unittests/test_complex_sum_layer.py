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
import paddle
from numpy.random import random as rand
from paddle import tensor
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
from paddle.fluid.framework import _test_eager_guard


class TestComplexSumLayer(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32", "float64"]
        self._places = [paddle.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_complex_basic_api(self):
        for dtype in self._dtypes:
            input = rand([2, 10, 10]).astype(dtype) + 1j * rand(
                [2, 10, 10]).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    var_x = dg.to_variable(input)
                    result = tensor.sum(var_x, axis=[1, 2]).numpy()
                    target = np.sum(input, axis=(1, 2))
                    self.assertTrue(np.allclose(result, target))

    def test_eager(self):
        with _test_eager_guard():
            self.test_complex_basic_api()


if __name__ == '__main__':
    unittest.main()
