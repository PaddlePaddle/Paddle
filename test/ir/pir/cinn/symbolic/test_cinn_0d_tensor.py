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
import sys
from os.path import dirname

import numpy as np

sys.path.append(dirname(dirname(__file__)))

import unittest

import utils

import paddle
import paddle.nn.functional as F
from paddle.static import InputSpec


class TestFunc(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()
        self.prepare_func()

    def prepare_data(self):
        pass

    def prepare_func(self):
        pass

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def check_output_shape(self, out):
        pass

    def eval_symbolic(self, use_cinn):
        paddle.seed(2024)
        func = utils.apply_to_static(self.func, use_cinn, self.input_spec)
        func.eval()
        out = func(*self.input)
        if use_cinn:
            self.check_jit_kernel_info(func)
            self.check_output_shape(out)
        return out

    def test_eval_symbolic(self):
        if type(self) is TestFunc:
            return
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), rtol=1e-6, atol=1e-3
        )


class TestReduce3Dto0D(TestFunc):
    def prepare_data(self):
        self.input_spec = [InputSpec(shape=[8, None, 64], dtype='float32')]
        self.input = [paddle.randn([8, 128, 64])]

    def prepare_func(self):
        def func(x):
            return paddle.sum(x)

        self.func = func

    def check_output_shape(self, out):
        np.testing.assert_equal(out.shape, ())


class TestReduce1Dto0D(TestReduce3Dto0D):
    def prepare_data(self):
        self.input_spec = [InputSpec(shape=[None], dtype='float32')]
        self.input = [paddle.randn([2048])]


class TestReduce0Dto0D(TestReduce3Dto0D):
    def prepare_data(self):
        self.input_spec = [InputSpec(shape=[], dtype='float32')]
        self.input = [paddle.randn([])]


class TestReduce3Dto0DThenRelu(TestReduce3Dto0D):
    def prepare_func(self):
        def func(x):
            return F.relu(paddle.sum(x))

        self.func = func


class TestReduce3Dto0DThenAdd0D(TestReduce3Dto0D):
    def prepare_data(self):
        self.input_spec = [
            InputSpec(shape=[8, None, 64], dtype='float32'),
            InputSpec(shape=[], dtype='float32'),
        ]
        self.input = [paddle.randn([8, 128, 64]), paddle.randn([])]

    def prepare_func(self):
        def func(x, y):
            return paddle.sum(x) + y

        self.func = func


class TestAdd0Dto3D(TestFunc):
    def prepare_data(self):
        self.input_spec = [
            InputSpec(shape=[], dtype='float32'),
            InputSpec(shape=[8, 128, 64], dtype='float32'),
        ]
        self.input = [paddle.randn([]), paddle.randn([8, 128, 64])]

    def prepare_func(self):
        def func(x, y):
            return x + y

        self.func = func


class TestAdd0Dto0D(TestAdd0Dto3D):
    def prepare_data(self):
        self.input_spec = [
            InputSpec(shape=[], dtype='float32'),
            InputSpec(shape=[], dtype='float32'),
        ]
        self.input = [paddle.randn([]), paddle.randn([])]

    def check_output_shape(self, out):
        np.testing.assert_equal(out.shape, ())


class TestSoftmax0D(TestReduce0Dto0D):
    def prepare_func(self):
        def func(x):
            x = paddle.exp(x)
            d = paddle.sum(x, axis=-1, keepdim=True)
            x = x / d
            return x

        self.func = func


class TestReshape0Dto3D(TestAdd0Dto3D):
    def prepare_func(self):
        def func(x, y):
            return paddle.reshape(x, [1, 1, 1]) + y

        self.func = func


class TestReshape0Dto0D(TestAdd0Dto0D):
    def prepare_func(self):
        def func(x, y):
            return paddle.reshape(x, []) + y

        self.func = func


class TestExpand0Dto3D(TestFunc):
    def prepare_data(self):
        self.input_spec = [InputSpec(shape=[], dtype='float32')]
        self.input = [paddle.randn([])]

    def prepare_func(self):
        def func(x):
            return paddle.expand(x, [8, 128, 64])

        self.func = func


class TestExpand0Dto0D(TestAdd0Dto0D):
    def prepare_func(self):
        def func(x, y):
            return paddle.expand(x, []) + y

        self.func = func


if __name__ == '__main__':
    unittest.main()
