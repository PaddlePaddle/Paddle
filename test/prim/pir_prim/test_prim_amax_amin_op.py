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
        self.op_name = None
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [10, 10, 10]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = None
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
            ops = [
                op.name()
                for op in fn.get_concrete_program(x)[-1]
                .program.backward_program.global_block()
                .ops
            ]
            assert self.op_name not in ops
            core._set_prim_all_enabled(False)
        return res, x_grad

    def test_prim_all(self):
        if self.net is None:
            return
        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.tol, atol=self.tol)


def amax_net1(x):
    return paddle.amax(x, keepdim=True)


def amax_net2(x):
    return paddle.amax(x, keepdim=False)


def amax_net3(x):
    return paddle.amax(x, axis=[0, 1], keepdim=False)


def amax_net4(x):
    return paddle.amax(x, axis=[-1, -2], keepdim=False)


def amax_net5(x):
    return paddle.amax(x, axis=[-1, 0], keepdim=False)


def amin_net1(x):
    return paddle.amin(x, keepdim=True)


def amin_net2(x):
    return paddle.amin(x, keepdim=False)


def amin_net3(x):
    return paddle.amin(x, axis=[0, 1], keepdim=False)


def amin_net4(x):
    return paddle.amin(x, axis=[-1, -2], keepdim=False)


def amin_net5(x):
    return paddle.amin(x, axis=[-1, 0], keepdim=False)


class TestPrimAmaxWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amax_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [10, 10, 10]
        self.x = np.ones(self.x_shape).astype(self.dtype)
        self.net = amax_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAmaxWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amax_grad"
        self.dtype = "float32"
        self.x_shape = [30]
        self.init_x_shape = [30]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amax_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAmaxWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amax_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [10, 10, 10]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amax_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAmaxWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amax_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [10, 10, 10]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amax_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAmaxWithGrad5(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amax_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [10, 10, 10]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.x[2] = self.x[4]
        self.net = amax_net4
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAmaxWithGrad6(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amax_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [10, 10, 10]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amax_net5
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAminWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amin_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [10, 10, 10]
        self.x = np.ones(self.x_shape).astype(self.dtype)
        self.net = amin_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAminWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amin_grad"
        self.dtype = "float32"
        self.x_shape = [30]
        self.init_x_shape = [30]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amin_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAminWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amin_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [10, 10, 10]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amin_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAminWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amin_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [10, 10, 10]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amin_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAminWithGrad5(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amin_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [10, 10, 10]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amin_net4
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAminWithGrad6(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amin_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 10]
        self.init_x_shape = [10, 10, 10]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.x[4] = self.x[7]
        self.net = amin_net5
        self.enable_cinn = False
        self.tol = 1e-6


if __name__ == "__main__":
    unittest.main()
