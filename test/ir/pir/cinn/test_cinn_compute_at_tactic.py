# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import utils

import paddle


class TestComputeAtTactic(unittest.TestCase):
    def eval(self, dy_compute, init_inputs):
        paddle.seed(2024)
        inputs = init_inputs()
        dy_out = dy_compute(*inputs)

        static_compute = utils.apply_to_static(dy_compute, use_cinn=True)
        st_out = static_compute(*inputs)

        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            np.testing.assert_allclose(a, b, atol=1e-3, rtol=1e-4)

    def test_multiple_reduce(self):
        def func(x, y):
            x0 = paddle.sum(
                x[:, :256]
                * (1.0 / (1.0 + paddle.exp(-1.0 * (x[:, :256] + y[256, 256]))))
            )
            x1 = paddle.sum(
                x[:, 256:]
                * (1.0 / (1.0 + paddle.exp(-1.0 * (x[:, 256:] + y[256, 256]))))
            )
            x2 = paddle.sum(
                x[:, :256]
                * (1.0 / (1.0 + paddle.exp(-1.0 * (x[:, :256] + y[128, 128]))))
            )
            x3 = paddle.sum(
                x[:, 256:]
                * (1.0 / (1.0 + paddle.exp(-1.0 * (x[:, 256:] + y[128, 128]))))
            )
            return x0, x1, x2, x3

        def init():
            x = paddle.randn([512, 512])
            y = paddle.randn([512, 512])
            return (x, y)

        self.eval(func, init)

    def test_reduce_with_condition(self):
        def func(a, b, c):
            a = paddle.sum(a)
            a = a / paddle.full([], 1.0)
            b = paddle.sum(b)
            b = b / paddle.full([], 1.0)
            c = paddle.sum(c)
            c = c / paddle.full([], 1.0)
            return a, b, c, a + b + c

        def init():
            a = paddle.randn([1])
            b = paddle.randn([1])
            c = paddle.randn([1])
            return (a, b, c)

        self.eval(func, init)


if __name__ == '__main__':
    unittest.main()
