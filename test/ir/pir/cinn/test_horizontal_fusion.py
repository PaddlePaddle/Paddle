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

import os
import unittest

import numpy
import utils

os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
os.environ['FLAGS_use_cinn'] = '1'

import paddle

build_strategy = paddle.static.BuildStrategy()
build_strategy.build_cinn_pass = True


def generate_input_spec(rank_dtype_list):
    input_spec = []
    for rank, dtype in rank_dtype_list:
        input_spec.append(
            paddle.static.InputSpec(shape=[None] * rank, dtype=dtype)
        )
    return input_spec


class TestHorizontalFusion(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def check_accuracy_and_kernel_num(
        self, data_init, dy_compute, kernel_num=None, input_spec=None
    ):
        inputs = data_init()
        dy_out = dy_compute(*inputs)
        static_compute = paddle.jit.to_static(
            full_graph=True,
            build_strategy=build_strategy,
            input_spec=input_spec,
        )(dy_compute)
        st_out = static_compute(*inputs)
        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            numpy.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
        if kernel_num is not None:
            utils.check_jit_kernel_number(static_compute, kernel_num)

    def test_trivial_trivial_without_shared_inputs(self):
        def func(x, y):
            x = paddle.sqrt(x + 1)
            y = paddle.exp(y / 2)
            return x, y

        def init():
            x = paddle.rand((32, 32, 128))
            y = paddle.rand((32, 1, 32, 128))
            return (x, y)

        self.check_accuracy_and_kernel_num(init, func, kernel_num=1)

    def test_trivial_reduce_without_shared_inputs(self):
        def func(x, y):
            x = paddle.sqrt(x + 1)
            y = paddle.sum(y / 2, axis=[1, 2], keepdim=True)
            return x, y

        def init():
            x = paddle.rand((128, 32, 16))
            y = paddle.rand((128, 32, 16))
            return (x, y)

        self.check_accuracy_and_kernel_num(init, func, kernel_num=1)

    def test_reduce_reduce_without_shared_inputs(self):
        def func(x, y):
            x = paddle.sqrt(paddle.mean(x + 1, axis=[1, 2]))
            y = paddle.sum(y / 2, axis=[1, 2], keepdim=True)
            return x, y

        def init():
            x = paddle.rand((128, 32, 16))
            y = paddle.rand((128, 32, 16))
            return (x, y)

        self.check_accuracy_and_kernel_num(init, func, kernel_num=1)

    def test_trivial_trivial_with_shared_inputs(self):
        def func(x):
            a = paddle.sqrt(paddle.transpose(x + 1, [1, 0, 2]))
            b = paddle.exp(paddle.transpose(x / 2, [2, 1, 0]))
            return a, b

        def init():
            x = paddle.rand((128, 32, 16))
            return (x,)

        self.check_accuracy_and_kernel_num(init, func, kernel_num=1)

    def test_trivial_reduce_with_shared_inputs(self):
        def func(x):
            a = paddle.sqrt(paddle.transpose(x + 1, [1, 0, 2]))
            b = paddle.exp(paddle.sum(x / 2, axis=[1, 2], keepdim=True))
            return a, b

        def init():
            x = paddle.rand((128, 32, 16))
            return (x,)

        self.check_accuracy_and_kernel_num(init, func, kernel_num=1)

    def test_reduce_reduce_with_shared_inputs(self):
        def func(x):
            a = paddle.max(paddle.transpose(x + 1, [2, 3, 0, 1]), axis=[0, 1])
            b = paddle.exp(paddle.sum(x / 2, axis=[2, 3], keepdim=True))
            return a, b

        def init():
            x = paddle.rand((4, 64, 32, 16))
            return (x,)

        self.check_accuracy_and_kernel_num(init, func, kernel_num=1)

    def test_cannot_fusion_because_of_memory_increase(self):
        def func(x, y, z, v):
            a = x + y
            b = z + v
            return a, b

        def init():
            x = paddle.rand((256, 512, 16, 16))
            y = paddle.rand((256, 512, 16, 16))
            z = paddle.rand((256, 512, 16, 16))
            v = paddle.rand((256, 512, 16, 16))
            return (x, y, z, v)

        self.check_accuracy_and_kernel_num(init, func, kernel_num=2)

    def test_reduce_horizontal_fusion(self):
        def func(x):
            a = paddle.sum(x, axis=[0], keepdim=True)
            a = paddle.reshape(a, [6])
            b = paddle.sum(x, axis=[0], keepdim=True)
            b = paddle.reshape(b, [6])
            return a, b

        def init():
            x = paddle.rand((2, 2, 3))
            return (x,)

        self.check_accuracy_and_kernel_num(init, func, kernel_num=1)

    def test_slice_concat_with_shared_input(self):
        def func(x):
            res = []
            for i in range(20):
                a = x[i : i + 1] + 1
                a = paddle.reshape(a, [1])
                axis = paddle.full(shape=[1], fill_value=i, dtype='int64')
                a = paddle.concat([a, axis], axis=0)
                res.append(a)
            return res

        def init():
            x = paddle.rand([20])
            x = paddle.cast(x, dtype='int64')
            return (x,)

        self.check_accuracy_and_kernel_num(init, func, kernel_num=1)


if __name__ == "__main__":
    unittest.main()
