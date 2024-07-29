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

import os
import unittest

import numpy as np
from numpy.random import random as rand

import paddle
import paddle.base.dygraph as dg
from paddle import base, tensor


class TestComplexSumLayer(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32", "float64"]
        self._places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            self._places.append(paddle.CPUPlace())
        if base.core.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_complex_basic_api(self):
        for dtype in self._dtypes:
            input = rand([2, 10, 10]).astype(dtype) + 1j * rand(
                [2, 10, 10]
            ).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    var_x = paddle.to_tensor(input)
                    result = tensor.sum(var_x, axis=[1, 2]).numpy()
                    target = np.sum(input, axis=(1, 2))
                    np.testing.assert_allclose(result, target, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
