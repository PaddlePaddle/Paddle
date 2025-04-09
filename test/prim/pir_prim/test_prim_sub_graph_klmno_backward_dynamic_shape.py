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
from test_prim_sub_graph_backward_dynamic_shape import (
    TestPrimBaseWithGrad,
    TestPrimThreeWithGrad,
    TestPrimTwoWithGrad,
)

import paddle


def kron_net(x, y):
    return paddle.kron(x, y)


def kthvalue_net1(x):
    return paddle.kthvalue(x, k=2)[0]


def kthvalue_net2(x):
    return paddle.kthvalue(x, k=1, axis=1)[0]


def kthvalue_net3(x):
    return paddle.kthvalue(x, k=1, axis=1, keepdim=True)[0]


def layer_norm_net1(x, scale=None, bias=None, epsilon=1e-05):
    input_shape = list(x.shape)
    normalized_shape = input_shape[1:]
    return paddle.nn.functional.layer_norm(
        x, normalized_shape, weight=scale, bias=bias, epsilon=epsilon
    )


def layer_norm_net2(x, scale=None, bias=None, epsilon=1e-05):
    input_shape = list(x.shape)
    normalized_shape = input_shape[2:]
    return paddle.nn.functional.layer_norm(
        x, normalized_shape, weight=scale, bias=bias, epsilon=epsilon
    )


def leaky_relu_net(x):
    return paddle.nn.functional.leaky_relu(x)


def logcumsumexp_net1(x):
    return paddle.logcumsumexp(x)


def logcumsumexp_net2(x):
    return paddle.logcumsumexp(x, axis=0)


def logcumsumexp_net3(x):
    return paddle.logcumsumexp(x, axis=-1)


def logsumexp_net1(x):
    return paddle.logsumexp(x)


def logsumexp_net2(x):
    return paddle.logsumexp(x, keepdim=False)


def logsumexp_net3(x):
    return paddle.logsumexp(x, axis=-1, keepdim=False)


def logsumexp_net4(x):
    return paddle.logsumexp(x, axis=[0, 2], keepdim=False)


def matmul_net(x, y):
    return paddle.matmul(x, y)


def max_net1(x):
    return paddle.max(x, keepdim=True)


def max_net2(x):
    return paddle.max(x, keepdim=False)


def max_net3(x):
    return paddle.max(x, axis=[0, 1], keepdim=False)


def max_net4(x):
    return paddle.max(x, axis=[-1, -2], keepdim=False)


def max_net5(x):
    return paddle.max(x, axis=[-1, 0], keepdim=False)


def max_net6(x):
    return paddle.max(x)


def maximum_net(x, y):
    return paddle.maximum(x, y)


def mean_net1(x):
    return paddle.mean(x, axis=1, keepdim=False)


def mean_net2(x):
    return paddle.mean(x, axis=-1, keepdim=False)


def mean_net3(x):
    return paddle.mean(x, axis=[0, 2], keepdim=False)


def minimum_net(x, y):
    return paddle.minimum(x, y)


def multiply_net(x, y):
    return x * y


class TestPrimKronWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.kron"
        self.dtype = "float32"
        self.x_shape = [10, 10]
        self.init_x_shape = [None, None]
        self.y_shape = [5, 5, 4]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = kron_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimKronWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.kron"
        self.dtype = "float32"
        self.x_shape = [10, 10]
        self.init_x_shape = [None, None]
        self.y_shape = [5, 5, 4, 3, 2]
        self.init_y_shape = [None, None, None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = kron_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimKronWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.kron"
        self.dtype = "float32"
        self.x_shape = [5, 5, 4, 3, 5, 6]
        self.init_x_shape = [None, None, None, None, None, None]
        self.y_shape = [3, 5, 4]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = kron_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimKthvalueWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.kthvalue_grad"
        self.dtype = "float32"
        self.x_shape = [30]
        self.init_x_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = kthvalue_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimKthvalueWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.kthvalue_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = kthvalue_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimKthvalueWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.kthvalue_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = kthvalue_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimKthvalueWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.kthvalue_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = kthvalue_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimLayerNormWithGrad1(TestPrimThreeWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.layer_norm_grad"
        self.dtype = "float32"
        self.x_shape = [20, 10, 60, 30]
        self.init_x_shape = [None, 10, None, None]
        self.y_shape = [10 * 60 * 30]
        self.init_y_shape = [None]
        self.z_shape = [10 * 60 * 30]
        self.init_z_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.z = np.random.random(self.z_shape).astype(self.dtype)
        self.net = layer_norm_net1
        self.enable_cinn = False
        self.tol = 1e-5
        self.other_place_tol = 1e-4

    def test_prim_all_dynamic(self):
        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        if not paddle.core.is_compiled_with_cuda():
            self.tol = self.other_place_tol

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.tol, atol=self.tol)


class TestPrimLayerNormWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.layer_norm_grad"
        self.dtype = "float32"
        self.x_shape = [20, 10, 60, 30]
        self.init_x_shape = [None, 10, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = layer_norm_net1
        self.enable_cinn = False
        self.tol = 1e-5
        self.other_place_tol = 1e-3

    def test_prim_all_dynamic(self):
        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        if not paddle.core.is_compiled_with_cuda():
            self.tol = self.other_place_tol

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.tol, atol=self.tol)


class TestPrimLayerNormWithGrad3(TestPrimLayerNormWithGrad1):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.layer_norm_grad"
        self.dtype = "float32"
        self.x_shape = [20, 10, 60, 70]
        self.init_x_shape = [None, 10, None, None]
        self.y_shape = [60 * 70]
        self.init_y_shape = [None]
        self.z_shape = [60 * 70]
        self.init_z_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.z = np.random.random(self.z_shape).astype(self.dtype)
        self.net = layer_norm_net2
        self.enable_cinn = False
        self.tol = 2e-5
        self.other_place_tol = 1e-4


class TestPrimLayerNormWithGrad4(TestPrimLayerNormWithGrad2):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.layer_norm_grad"
        self.dtype = "float32"
        self.x_shape = [20, 10, 60, 70]
        self.init_x_shape = [None, 10, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = layer_norm_net1
        self.enable_cinn = False
        self.tol = 1e-5
        self.other_place_tol = 8e-3


class TestPrimLeakyReluWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.leaky_relu_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = leaky_relu_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimLogcumsumexpWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.logcumsumexp_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = logcumsumexp_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimLogcumsumexpWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.logcumsumexp_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = logcumsumexp_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimLogcumsumexpWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.logcumsumexp_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = logcumsumexp_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimLogsumexpWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.logsumexp_grad"
        self.dtype = "float32"
        self.x_shape = [1000]
        self.init_x_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = logsumexp_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimLogsumexpWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.logsumexp_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = logsumexp_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimLogsumexpWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.logsumexp_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = logsumexp_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimLogsumexpWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.logsumexp_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = logsumexp_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimLogsumexpWithGrad5(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.logsumexp_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = logsumexp_net4
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMatmulWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.matmul_grad"
        self.dtype = "float32"
        self.x_shape = [30, 40, 200]
        self.init_x_shape = [None, None, 200]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = matmul_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMatmulWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.matmul_grad"
        self.dtype = "float32"
        self.x_shape = [1, 30, 40, 200]
        self.init_x_shape = [None, None, None, 200]
        self.y_shape = [30, 1, 200, 40]
        self.init_y_shape = [None, None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = matmul_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMatmulWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.matmul_grad"
        self.dtype = "float32"
        self.x_shape = [1, 30, 40, 200]
        self.init_x_shape = [1, None, None, 200]
        self.y_shape = [30, 1, 200, 40]
        self.init_y_shape = [None, 1, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = matmul_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMatmulWithGrad4(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.matmul_grad"
        self.dtype = "float32"
        self.x_shape = [30, 1, 40, 200]
        self.init_x_shape = [None, None, None, 200]
        self.y_shape = [1, 30, 200, 40]
        self.init_y_shape = [None, None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = matmul_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMatmulWithGrad5(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.matmul_grad"
        self.dtype = "float32"
        self.x_shape = [30, 1, 40, 200]
        self.init_x_shape = [None, 1, None, 200]
        self.y_shape = [1, 30, 200, 40]
        self.init_y_shape = [1, None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = matmul_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMaxWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.max_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = max_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMaxWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.max_grad"
        self.dtype = "float32"
        self.x_shape = [30]
        self.init_x_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = max_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMaxWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.max_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = max_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMaxWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.max_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = max_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMaxWithGrad5(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.max_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = max_net4
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMaxWithGrad6(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.max_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = max_net5
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMaxWithGrad7(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.max_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = max_net6
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMaximumWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.maximum_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.y_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = maximum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMaximumWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.maximum_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.y_shape = [200, 40]
        self.init_x_shape = [None, None, None]
        self.init_y_shape = [None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = maximum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMaximumWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.maximum_grad"
        self.dtype = "float32"
        self.x_shape = [200, 40]
        self.y_shape = [30, 200, 40]
        self.init_x_shape = [None, None]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = maximum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMaximumWithGrad4(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.maximum_grad"
        self.dtype = "float32"
        self.x_shape = [40]
        self.y_shape = [30, 200, 40]
        self.init_x_shape = [None]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = maximum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMaximumWithGrad5(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.maximum_grad"
        self.dtype = "float32"
        self.x_shape = [1, 1]
        self.y_shape = [30, 200, 40]
        self.init_x_shape = [None, None]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = maximum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMaximumWithGrad6(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.maximum_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.y_shape = [1, 1]
        self.init_x_shape = [None, None, None]
        self.init_y_shape = [None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = maximum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMeanWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.mean_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = mean_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMeanWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.mean_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = mean_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMeanWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.mean_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = mean_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMinimumWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.minimum_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.y_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = minimum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMinimumWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.minimum_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.y_shape = [200, 40]
        self.init_x_shape = [None, None, None]
        self.init_y_shape = [None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = minimum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMinimumWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.minimum_grad"
        self.dtype = "float32"
        self.x_shape = [200, 40]
        self.y_shape = [30, 200, 40]
        self.init_x_shape = [None, None]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = minimum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMinimumWithGrad4(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.minimum_grad"
        self.dtype = "float32"
        self.x_shape = [40]
        self.y_shape = [30, 200, 40]
        self.init_x_shape = [None]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = minimum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMinimumWithGrad5(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.minimum_grad"
        self.dtype = "float32"
        self.x_shape = [1, 1]
        self.y_shape = [30, 200, 40]
        self.init_x_shape = [None, None]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = minimum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMinimumWithGrad6(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.minimum_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.y_shape = [1, 1]
        self.init_x_shape = [None, None, None]
        self.init_y_shape = [None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = minimum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMultiplyWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.multiply_grad"
        self.dtype = "float32"
        self.x_shape = [1, 1, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = multiply_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMultiplyWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.multiply_grad"
        self.dtype = "float32"
        self.x_shape = [1, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = multiply_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMultiplyWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.multiply_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = multiply_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMultiplyWithGrad4(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.multiply_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [1, 1, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = multiply_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMultiplyWithGrad5(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.multiply_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [1, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = multiply_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMultiplyWithGrad6(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.multiply_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = multiply_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMultiplyWithGrad7(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.multiply_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = multiply_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMultiplyWithGrad8(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.multiply_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [40]
        self.init_y_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = multiply_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMultiplyWithGrad9(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.multiply_grad"
        self.dtype = "float32"
        self.x_shape = [40]
        self.init_x_shape = [None]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = multiply_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMultiplyWithGrad10(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.multiply_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.y_shape = [200, 40]
        self.init_y_shape = self.y_shape
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = multiply_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimMultiplyWithGrad11(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.multiply_grad"
        self.dtype = "float32"
        self.x_shape = [200, 40]
        self.init_x_shape = self.x_shape
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = multiply_net
        self.enable_cinn = False
        self.tol = 1e-5


if __name__ == "__main__":
    unittest.main()
