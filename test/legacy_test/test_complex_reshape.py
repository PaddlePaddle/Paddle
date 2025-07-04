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

import paddle
import paddle.base.dygraph as dg
from paddle import base


class TestComplexReshape(unittest.TestCase):
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

    def test_shape_norm_dims(self):
        for dtype in self._dtypes:
            x_np = np.random.randn(2, 3, 4).astype(
                dtype
            ) + 1j * np.random.randn(2, 3, 4).astype(dtype)
            shape = (2, -1)
            for place in self._places:
                with dg.guard(place):
                    x_var = paddle.to_tensor(x_np)
                    y_var = paddle.reshape(x_var, shape)
                    y_np = y_var.numpy()
                    np.testing.assert_allclose(
                        np.reshape(x_np, shape), y_np, rtol=1e-05
                    )

    def test_shape_omit_dims(self):
        for dtype in self._dtypes:
            x_np = np.random.randn(2, 3, 4).astype(
                dtype
            ) + 1j * np.random.randn(2, 3, 4).astype(dtype)
            shape = (0, -1)
            shape_ = (2, 12)
            for place in self._places:
                with dg.guard(place):
                    x_var = paddle.to_tensor(x_np)
                    y_var = paddle.reshape(x_var, shape)
                    y_np = y_var.numpy()
                    np.testing.assert_allclose(
                        np.reshape(x_np, shape_), y_np, rtol=1e-05
                    )


if __name__ == "__main__":
    unittest.main()
