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


class TestArange(unittest.TestCase):
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

    def test_arange_postfuse(self):
        # post fusion
        def func():
            return paddle.arange(0, 1024, 2.1, dtype="float32") * 0.2 + 1

        self.eval(func, [])

    def test_arange_cast(self):
        # arange cast (IR node type)
        def func():
            return (
                paddle.arange(0, 2048, 0.25, dtype="float64")
                .min()
                .astype("int32")
            )

        self.eval(func, [])

    def test_default_hv_fusing(self):
        def func():
            # kwargs
            tensor = paddle.arange(start=1024, end=0, step=-4, dtype="int32")
            return paddle.arange(256).astype("int32") + tensor, paddle.arange(
                end=512, step=2
            )

        self.eval(func, [])

    def test_broadcast_tensor(self):
        def func():
            a = paddle.arange(1, 2, dtype="int32")
            b = paddle.arange(end=32769)
            return (a * 3).astype("int64") + b

        self.eval(func, [])

    def test_arange_slice_reshape_index(self):
        # tensor slicing, indexing and reshaping
        def func(x):
            indices = paddle.arange(2048, dtype="int64")[:1024]
            return x[indices, :1].reshape([1, -1])

        x_spec = InputSpec(shape=[1024, 64])
        x = paddle.randn([1024, 64])
        self.eval(func, [x], [x_spec])

    def test_arange_dynamic(self):
        # cinn_op.generate_shape symbolic input (abs needed)
        def func(x, y):
            stop = paddle.shape(y)[1] - 3
            start = paddle.shape(x)[0] * 2
            return paddle.arange(start, stop, 2, dtype="int64")

        x_spec = InputSpec(shape=[-1, 2, 3], dtype="int64")
        y_spec = InputSpec(shape=[2, -1, 3], dtype="int64")

        x = paddle.zeros([2, 2, 3], dtype="int64")
        y = paddle.zeros([2, 11, 3], dtype="int64")
        self.eval(func, [x, y], [x_spec, y_spec])
        self.eval(func, [x, y], None)

    def test_arange_dynamic_nested_broadcast(self):
        # nested cinn_op.arange and generate_shape with broadcast
        def func(x, y):
            start = paddle.shape(x)[0]
            stop = paddle.shape(y)[1]
            res = paddle.arange(start, stop, 2, dtype="int32")
            return (
                paddle.arange(0, res.shape[0], 1, dtype="int32") * 2
                + res
                + paddle.ones([res.shape[0], 1], dtype="int32")
            )

        x_spec = InputSpec(shape=[-1, 16], dtype="int32")
        y_spec = InputSpec(shape=[16, -1], dtype="int32")

        x = paddle.zeros([2, 16], dtype="int32")
        y = paddle.zeros([16, 32], dtype="int32")
        self.eval(func, [x, y], [x_spec, y_spec])
        self.eval(func, [x, y], None)


if __name__ == "__main__":
    unittest.main()
