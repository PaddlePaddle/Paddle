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


class Test0Size(unittest.TestCase):
    def eval(self, dy_compute, inputs):
        dy_out = dy_compute(*inputs)

        static_compute = utils.apply_to_static(dy_compute, use_cinn=True)
        st_out = static_compute(*inputs)

        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            numpy.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)

    def test_r_r0(self):
        def func(x):
            return x.sum()

        x = paddle.uniform([128, 0])

        self.eval(func, [x])

    def test_s0_s(self):
        def func(x, y):
            return x + y

        x = paddle.uniform([0, 128])
        y = paddle.uniform([128])

        self.eval(func, [x, y])

    def test_s0_r(self):
        def func(x):
            return x.sum(axis=1)

        x = paddle.uniform([0, 128])

        self.eval(func, [x])

    def test_r_r0_s(self):
        def func(x):
            return x.sum(axis=(0, 1))

        x = paddle.uniform([32, 0, 128])

        self.eval(func, [x])

    def test_s_r_s0(self):
        def func(x):
            return x.sum(axis=1)

        x = paddle.uniform([16, 32, 0])

        self.eval(func, [x])


if __name__ == "__main__":
    unittest.main()
