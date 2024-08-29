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
    TestPrimTwoWithGrad,
    apply_to_static,
)

import paddle
from paddle.framework import core
from paddle.static import InputSpec


def add_net(x, y):
    return x + y


def concat_net1(x):
    y = x + 1
    return paddle.concat([x, y], axis=-1)


def concat_net2(x):
    y = x + 1
    return paddle.concat([x, y], axis=1)


def concat_net3(x):
    return paddle.concat(x, axis=0)


def cumsum_net(x):
    return paddle.cumsum(x, axis=1)


def divide_net(x, y):
    return x / y


def dropout_net1(x):
    return paddle.nn.functional.dropout(
        x, training=False, mode='downscale_in_infer'
    )


def dropout_net2(x):
    return paddle.nn.functional.dropout(x)


def dropout_net3(x):
    return paddle.nn.functional.dropout(x, 1.0)


def elementwise_pow_net(x, y):
    return paddle.pow(x, y)


def erf_net(x):
    return paddle.erf(x)


def expand_net(x):
    return paddle.expand(x, [30, 200, 40])


class TestPrimAddWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 1, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = add_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAddWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = add_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAddWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = add_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAddWithGrad4(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [1, 1, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = add_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAddWithGrad5(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [1, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = add_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAddWithGrad6(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = add_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAddWithGrad7(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [40]
        self.init_y_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = add_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAddWithGrad8(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [40]
        self.init_x_shape = [None]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = add_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAddWithGrad9(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.y_shape = [200, 40]
        self.init_y_shape = self.y_shape
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = add_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAddWithGrad10(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [200, 40]
        self.init_x_shape = self.x_shape
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = add_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimConcatWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = concat_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimConcatWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = concat_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimConcatWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = concat_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimConcatWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, 200, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = concat_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimConcatWithGrad5(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        x = np.random.random(self.x_shape).astype(self.dtype)
        self.x = [x + i for i in range(4)]
        self.net = concat_net3
        self.enable_cinn = False
        self.tol = 1e-6

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_all_enabled(True)
        x = [paddle.to_tensor(self.x[i], stop_gradient=False) for i in range(4)]
        if flag == "prim":
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    [
                        InputSpec(shape=self.x_shape, dtype='float32'),
                        InputSpec(shape=self.init_x_shape, dtype='float32'),
                        InputSpec(shape=self.init_x_shape, dtype='float32'),
                        InputSpec(shape=self.x_shape, dtype='float32'),
                    ]
                ],
            )
            fn.train()
        else:
            fn = self.net
        res = fn(x)
        res.backward()
        x_grad1 = x[0].gradient()
        x_grad2 = x[1].gradient()
        x_grad3 = x[2].gradient()
        x_grad4 = x[3].gradient()
        if flag == "prim":
            core._set_prim_all_enabled(False)
        return res, [x_grad1, x_grad2, x_grad3, x_grad4]

    def test_prim_all_dynamic(self):
        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.tol, atol=self.tol)


class TestPrimConcatWithGrad6(TestPrimConcatWithGrad5):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, 200, None]
        x = np.random.random(self.x_shape).astype(self.dtype)
        self.x = [x + i for i in range(4)]
        self.net = concat_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimCumsumWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = cumsum_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimDivideWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 1, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = divide_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimDivideWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = divide_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimDivideWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = divide_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimDivideWithGrad4(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [1, 1, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = divide_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimDivideWithGrad5(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [1, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = divide_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimDivideWithGrad6(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = divide_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimDivideWithGrad7(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = divide_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimDivideWithGrad8(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [40]
        self.init_y_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = divide_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimDivideWithGrad9(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [40]
        self.init_x_shape = [None]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = divide_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimDivideWithGrad10(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.y_shape = [200, 40]
        self.init_y_shape = self.y_shape
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = divide_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimDivideWithGrad11(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [200, 40]
        self.init_x_shape = self.x_shape
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = divide_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimDropoutWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 40, 50]
        self.init_x_shape = [None, None, 50]
        self.x = np.ones(self.x_shape).astype(self.dtype)
        self.net = dropout_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimDropoutWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [300, 4096]
        self.init_x_shape = [None, 4096]
        self.x = np.ones(self.x_shape).astype(self.dtype)
        self.net = dropout_net2
        self.enable_cinn = False
        self.tol = 0.07

    def test_prim_all_dynamic(self):
        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref.sum(), actual.sum(), rtol=self.tol, atol=self.tol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(
                dr.sum(), d.sum(), rtol=self.tol, atol=self.tol
            )


class TestPrimDropoutWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 40, 50]
        self.init_x_shape = [None, None, 50]
        self.x = np.ones(self.x_shape).astype(self.dtype)
        self.net = dropout_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimElementwisePowWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 1, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = elementwise_pow_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimElementwisePowWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = elementwise_pow_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimElementwisePowWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = elementwise_pow_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimElementwisePowWithGrad4(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [1, 1, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = elementwise_pow_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimElementwisePowWithGrad5(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [1, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = elementwise_pow_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimElementwisePowWithGrad6(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = elementwise_pow_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimElementwisePowWithGrad7(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = elementwise_pow_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimElementwisePowWithGrad8(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [40]
        self.init_y_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = elementwise_pow_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimElementwisePowWithGrad9(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [40]
        self.init_x_shape = [None]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = elementwise_pow_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimElementwisePowWithGrad10(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.y_shape = [200, 40]
        self.init_y_shape = self.y_shape
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = elementwise_pow_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimElementwisePowWithGrad11(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [200, 40]
        self.init_x_shape = self.x_shape
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = elementwise_pow_net
        self.enable_cinn = False
        self.tol = 1e-5


class TestPrimErfWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = erf_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimExpandWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [200, 40]
        self.init_x_shape = [None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = expand_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimExpandWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 1, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = expand_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimExpandWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = expand_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimExpandWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 1, 1]
        self.init_x_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = expand_net
        self.enable_cinn = False
        self.tol = 1e-6


if __name__ == "__main__":
    unittest.main()
