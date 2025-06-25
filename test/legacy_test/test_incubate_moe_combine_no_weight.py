#  Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import unittest

import numpy as np

import paddle
import paddle.incubate.nn.functional as F


def gen_test_case(S, K, Dim, seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    x_numpy = np.random.rand(S * K, Dim).astype(np.float32)
    scatter_index_numpy = np.random.permutation(max(x_numpy.shape[0], S * K))[
        : S * K
    ].astype("int64")
    scatter_index_numpy = scatter_index_numpy.reshape([S, K])
    scatter_index_numpy[scatter_index_numpy >= x_numpy.shape[0]] = 0
    grad_numpy = np.random.randn(S, Dim).astype(np.float32)
    return x_numpy, scatter_index_numpy, grad_numpy


def moe_combine_no_weight_ref(x, scatter_index):
    return paddle.sum(x[scatter_index], axis=1)


class TestCombineFwd(unittest.TestCase):
    def setUp(self):
        self.s = 4096
        self.k = 8
        self.h = 8192
        self.dtype = "float32"
        self.rtol = 1e-05
        self.atol = 1e-05

    def test_forward_backward(self):
        x_numpy, scatter_index_numpy, grad_numpy = gen_test_case(
            self.s, self.k, self.h
        )

        scatter_index = paddle.to_tensor(scatter_index_numpy, "int32")
        y_grad = paddle.to_tensor(grad_numpy, self.dtype)

        # Compute reference y_1 and x_grad_1
        x_1 = paddle.to_tensor(x_numpy, self.dtype)
        x_1.stop_gradient = False

        y_1 = moe_combine_no_weight_ref(x_1, scatter_index)
        paddle.autograd.backward(tensors=[y_1], grad_tensors=[y_grad])

        y_1 = y_1.cast("float32")
        x_grad_1 = x_1.grad.cast("float32")

        # Compute target y_2 and x_grad_2
        x_2 = paddle.to_tensor(x_numpy, self.dtype)
        x_2.stop_gradient = False

        combine_weight = paddle.ones([self.s, self.k], dtype=self.dtype)
        y_2 = F.moe_combine_no_weight(x_2, combine_weight, scatter_index)
        paddle.autograd.backward(tensors=[y_2], grad_tensors=[y_grad])

        y_2 = y_2.cast("float32")
        x_grad_2 = x_2.grad.cast("float32")

        # Compare y and x_grad
        np.testing.assert_allclose(y_1, y_2, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(
            x_grad_1, x_grad_2, rtol=self.rtol, atol=self.atol
        )


class TestCombineFwd1(TestCombineFwd):
    def setUp(self):
        self.s = 1024
        self.k = 8
        self.h = 1024
        self.dtype = "float32"
        self.rtol = 1e-5
        self.atol = 1e-5


class TestCombineFwd2(TestCombineFwd):
    def setUp(self):
        self.s = 1024
        self.k = 8
        self.h = 1024
        self.dtype = "bfloat16"
        self.rtol = 1
        self.atol = 10


class TestCombineFwd3(TestCombineFwd):
    def setUp(self):
        self.s = 100
        self.k = 2
        self.h = 8192
        self.dtype = "bfloat16"
        self.rtol = 1
        self.atol = 1


class TestCombineFwd4(TestCombineFwd):
    def setUp(self):
        self.s = 100
        self.k = 16
        self.h = 300
        self.dtype = "bfloat16"
        self.rtol = 1e-2
        self.atol = 10


if __name__ == "__main__":
    unittest.main()
