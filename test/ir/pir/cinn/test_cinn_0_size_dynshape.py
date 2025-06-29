# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy
import utils

import paddle
from paddle.static import InputSpec


class Test0SizeDynShape(unittest.TestCase):
    def eval(self, dy_compute, inputs, input_spec=None):
        dy_out = dy_compute(*inputs)

        static_compute = utils.apply_to_static(
            dy_compute, use_cinn=True, input_spec=input_spec
        )
        st_out = static_compute(*inputs)

        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            numpy.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)

    def test_s0_s_dynshape(self):
        def func(x, y):
            return x + y

        x = paddle.uniform([0, 128])
        x_spec = InputSpec([None, 128])
        y = paddle.uniform([1, 128])
        y_spec = InputSpec([None, 128])

        self.eval(func, [x, y], [x_spec, y_spec])

    def test_s0_r_dynshape(self):
        def func(x):
            return x.sum(axis=1)

        x = paddle.uniform([0, 128])
        x_spec = InputSpec([0, None])

        self.eval(func, [x], [x_spec])

    def test_r0_s_dynshape(self):
        def func(x):
            return x.sum(axis=0)

        x = paddle.uniform([0, 128])
        x_spec = InputSpec([None, 128])

        self.eval(func, [x], [x_spec])

    def test_r_s_r0_dynshape(self):
        def func(x):
            return x.sum(axis=(0, 2))

        x = paddle.uniform([16, 32, 0])
        x_spec = InputSpec([16, None, 0])

        self.eval(func, [x], [x_spec])


if __name__ == "__main__":
    unittest.main()
