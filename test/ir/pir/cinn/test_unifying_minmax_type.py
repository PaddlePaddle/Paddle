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


class TestMinMaxOperandCast:
    def eval(self, dy_compute, inputs, input_spec=None):
        dy_out = dy_compute(*inputs)

        static_compute = utils.apply_to_static(
            dy_compute, input_spec=input_spec, use_cinn=True
        )
        st_out = static_compute(*inputs)

        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            numpy.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)

    def test_simplest(self):
        def func(x):
            y = paddle.zeros([1024], dtype="int32")
            return paddle.minimum(
                y, paddle.full([1024], x.shape[1], dtype="int32")
            )

        x_spec = InputSpec(shape=[1024, -1])
        x = paddle.randn([1024, 64])
        self.eval(func, [x], [x_spec])

    def test_fuse_with_gather_nd(self):
        def func(x):
            indices = paddle.full([2048], x.shape[1], dtype="int32")[:1024]
            return x[indices, :1].reshape([1, -1])

        x_spec = InputSpec(shape=[1024, -1])
        x = paddle.randn([1024, 64])
        self.eval(func, [x], [x_spec])

    def test_nested_min_max(self):
        def func(x, y, z):
            maximum = paddle.maximum(
                z, paddle.full([x.shape[0]], 2 * x.shape[1] + 1, dtype="int32")
            )
            return paddle.minimum(y[:512], maximum)

        x_spec = InputSpec(shape=[-1, -1])
        y_spec = InputSpec(shape=[1024])
        z_spec = InputSpec(shape=[-1])
        x = paddle.randn([512, 90])
        y = paddle.randint(0, 128, [1024], dtype="int32")
        z = paddle.randint(0, 128, [x.shape[0]], dtype="int32")
        self.eval(func, [x, y, z], [x_spec, y_spec, z_spec])

    def test_gather_nd_slice(self):
        def func(x):
            indices = paddle.full([2048], x.shape[1], dtype="int32")[:1024]
            return x[indices, :1].reshape([1, -1])

        x_spec = InputSpec(shape=[1024, -1])
        x = paddle.randn([1024, 64])
        self.eval(func, [x], [x_spec])

    def test_multi_dim_reduce(self):
        def func(x, y):
            max_threshold = paddle.full(
                x.shape, x.shape[1] * x.shape[1] - 20, dtype="int32"
            )
            min_threshold = paddle.full(x.shape[:-1], x.shape[1], dtype="int64")
            min_vals = paddle.minimum(max_threshold, x + y)
            reduced = min_vals.max(axis=-1).to("int64")
            return paddle.maximum(reduced, min_threshold).sum(axis=0).max()

        x_spec = InputSpec(shape=[99, -1, 18])
        y_spec = InputSpec(shape=[99, -1, 1])
        x = paddle.randint(0, 128, [99, 17, 18], dtype="int32")
        y = paddle.randint(0, 128, [99, 17, 1], dtype="int32")
        self.eval(func, [x, y], [x_spec, y_spec])


if __name__ == "__main__":
    unittest.main()
