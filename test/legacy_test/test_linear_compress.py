# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import fluid
from paddle.fluid.framework import default_main_program
from paddle.framework import set_default_dtype

np.random.seed(123)
paddle.seed(123)
default_main_program().random_seed = 42
paddle.disable_static()


class LinearTestCase(unittest.TestCase):
    def config(self):
        self.dtype = 'float16'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.bias = True
        self.in_features = 64
        self.out_features = 64
        self.algo = "weight_only"
        self.bits = 8

    def setUp(self):
        self.config()
        input = np.random.random((2, 4, self.in_features))
        self.input = paddle.to_tensor(input, dtype=self.dtype)
        if self.bias:
            bias_attr = fluid.ParamAttr(
                learning_rate=0.8,
                trainable=False,
                regularizer=None,
                initializer=paddle.nn.initializer.Constant(value=1.0),
            )
        else:
            bias_attr = None
        set_default_dtype(self.dtype)
        self.linear = paddle.nn.Linear(
            self.in_features, self.out_features, bias_attr=bias_attr
        )
        if self.algo == "llm.int8":
            self.config = {"threshold": 6.0}
        else:
            self.config = None
        self.linear_compress = paddle.nn.LinearCompress(
            self.in_features,
            self.out_features,
            bias_attr=bias_attr,
            bits=8,
            algo=self.algo,
            config=self.config,
        )
        self.linear_compress(self.input)

    def get_linear_out(self):
        out = self.linear(self.input)
        return out.numpy()

    def get_linear_compress_out(self):
        out = self.linear_compress(self.input)
        return out.numpy()

    def test_linear_compress(self):
        out_real = self.get_linear_compress_out()
        out_expect = self.get_linear_out()
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


class LinearTestCase1(LinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = True
        self.in_features = 128
        self.out_features = 64


class LinearTestCase2(LinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = False
        self.in_features = 64
        self.out_features = 64


class LinearTestCase3(LinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = False
        self.in_features = 64
        self.out_features = 64
        self.algo = "llm.int8"
        self.atol = 1e-1


class LinearTestCase4(LinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = True
        self.in_features = 128
        self.out_features = 64
        self.bits = 4


if __name__ == '__main__':
    unittest.main()
