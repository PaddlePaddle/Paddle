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
from test_prim_sub_graph_backward_dynamic_shape import TestPrimBaseWithGrad

import paddle
from paddle.framework import core


def floor_net(x):
    return paddle.floor(x)


def gelu_net1(x):
    return paddle.nn.functional.gelu(x, approximate=True)


def gelu_net2(x):
    return paddle.nn.functional.gelu(x, approximate=False)


def hardswish_net(x):
    return paddle.nn.functional.hardswish(x)


class TestPrimFloorWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = floor_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimGeluWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = gelu_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimGeluWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = gelu_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimGeluWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.dtype = "float16"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.uniform(-1, 1, size=self.x_shape).astype(self.dtype)
        self.net = gelu_net1
        self.enable_cinn = False
        self.rtol = 1e-5
        self.atol = 0.0005

    def test_prim_all_dynamic(self):
        if not paddle.is_compiled_with_cuda():
            return
        place = core.CUDAPlace(0)
        if not core.is_float16_supported(place):
            return

        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.rtol, atol=self.atol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.rtol, atol=self.atol)


class TestPrimGeluWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.dtype = "float16"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.uniform(-1, 1, size=self.x_shape).astype(self.dtype)
        self.net = gelu_net2
        self.enable_cinn = False
        self.rtol = 1e-5
        self.atol = 0.0005

    def test_prim_all_dynamic(self):
        if not paddle.is_compiled_with_cuda():
            return
        place = core.CUDAPlace(0)
        if not core.is_float16_supported(place):
            return

        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.rtol, atol=self.atol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.rtol, atol=self.atol)


class TestPrimHardswishWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = hardswish_net
        self.enable_cinn = False
        self.tol = 1e-6


if __name__ == "__main__":
    unittest.main()
