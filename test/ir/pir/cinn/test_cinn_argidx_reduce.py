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


class TestArgIdxReduce(unittest.TestCase):
    def eval(self, dy_compute, inputs, input_spec=None):
        dy_out = dy_compute(*inputs)

        static_compute = utils.apply_to_static(
            dy_compute, use_cinn=True, input_spec=None
        )
        st_out = static_compute(*inputs)

        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            numpy.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)

    def test_argmax_argmin_add(self):
        # block reduce
        def func(x):
            return x.argmax(axis=1) + x.argmin(axis=1)

        x = paddle.randn([128, 256])
        self.eval(func, [x])

    def test_argmax_discrete(self):
        # discrete reduce
        def func(x):
            return x[x.argmax(axis=0, keepdim=True), paddle.arange(x.shape[-1])]

        x = paddle.randn([66, 256])
        self.eval(func, [x])

    def test_argmin_all_bc(self):
        # reduce axis = None case
        def func(x, y):
            return y - x.argmin(dtype='int32')

        x = paddle.randn([64, 96, 32])
        y = paddle.randint(0, 0xFFFF, [32, 64, 32], dtype='int32')
        self.eval(func, [x, y])

    def test_dynamic_shape(self):
        def func(x):
            return x.reshape([8, -1]).argmin(axis=-1)

        x = paddle.randn([8, 17, 19])
        x_spec = InputSpec(shape=[8, None, None])
        self.eval(func, [x], [x_spec])

    def test_slice_grid_composite(self):
        # tensor slicing, grid reduce, func composition
        def func(x):
            return x[..., 313:].reshape([-1, 32]).argmin(axis=0).argmax(axis=0)

        x_spec = InputSpec(shape=[32, None, 8500])
        x = paddle.randn([32, 2, 8500])
        self.eval(func, [x], [x_spec])


if __name__ == "__main__":
    unittest.main()
