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

import paddle
from paddle.framework import core
from paddle.static import InputSpec


def sum_net1(x):
    return paddle.sum(x, axis=1, keepdim=False)


def sum_net2(x):
    return paddle.sum(x)


def sum_net3(x):
    return paddle.sum(x, keepdim=True)


def sum_net4(x):
    return paddle.sum(x, axis=-1, keepdim=False)


def sum_net5(x):
    return paddle.sum(x, axis=[0, 2], keepdim=False)


def mean_net1(x):
    return paddle.mean(x, axis=1, keepdim=False)


def mean_net2(x):
    return paddle.mean(x, axis=-1, keepdim=False)


def mean_net3(x):
    return paddle.mean(x, axis=[0, 2], keepdim=False)


def add_net(x, y):
    return x + y


def subtract_net(x, y):
    return x - y


def concat_net1(x):
    y = x + 1
    return paddle.concat([x, y], axis=-1)


def concat_net2(x):
    y = x + 1
    return paddle.concat([x, y], axis=1)


def split_net1(x):
    res = paddle.split(x, num_or_sections=10, axis=-1)
    tmp_res = res[0]
    for i in range(1, len(res)):
        tmp_res = tmp_res + res[i] * i
    return tmp_res / len(res)


def split_net2(x):
    res = paddle.split(x, num_or_sections=10, axis=1)
    tmp_res = res[0]
    for i in range(1, len(res)):
        tmp_res = tmp_res + res[i] * i
    return tmp_res / len(res)


def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )


class TestPrimBaseWithGrad(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sum_net1
        self.enable_cinn = False
        self.tol = 1e-6

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_all_enabled(True)
        x = paddle.to_tensor(self.x, stop_gradient=False)
        if flag == "prim":
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    InputSpec(shape=self.init_x_shape, dtype='float32'),
                ],
            )
            fn.train()
        else:
            fn = self.net
        res = fn(x)
        res.backward()
        x_grad = x.gradient()
        if flag == "prim":
            core._set_prim_all_enabled(False)
        return res, x_grad

    def test_prim_all_dynamic(self):
        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.tol, atol=self.tol)


class TestPrimSumWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1000]
        self.init_x_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sum_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSumWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sum_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSumWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sum_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSumWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sum_net4
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSumWithGrad5(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sum_net5
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMeanWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
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
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = mean_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimTwoWithGrad(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = add_net
        self.enable_cinn = False
        self.tol = 1e-6

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_all_enabled(True)
        x = paddle.to_tensor(self.x, stop_gradient=False)
        y = paddle.to_tensor(self.y, stop_gradient=False)
        if flag == "prim":
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    InputSpec(shape=self.init_x_shape, dtype='float32'),
                    InputSpec(shape=self.init_y_shape, dtype='float32'),
                ],
            )
            fn.train()
        else:
            fn = self.net
        res = fn(x, y)
        res.backward()
        x_grad = x.gradient()
        y_grad = y.gradient()
        if flag == "prim":
            core._set_prim_all_enabled(False)
        return res, [x_grad, y_grad]

    def test_prim_all_dynamic(self):
        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.tol, atol=self.tol)


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


class TestPrimSubtractWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 1, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad4(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [1, 1, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad5(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [1, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad6(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad7(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad8(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [40]
        self.init_y_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad9(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [40]
        self.init_x_shape = [None]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
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


class TestPrimSplitWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, 200, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = split_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSplitWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = split_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSplitWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = split_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSplitWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, 200, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = split_net2
        self.enable_cinn = False
        self.tol = 1e-6


if __name__ == "__main__":
    unittest.main()
