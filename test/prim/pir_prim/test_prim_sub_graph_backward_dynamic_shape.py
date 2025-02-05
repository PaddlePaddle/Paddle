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


def add_net(x, y):
    return x + y


def batch_norm_net1(x, y, z):
    mean = paddle.zeros([40], dtype="float32")
    var = paddle.ones([40], dtype='float32')
    return paddle.nn.functional.batch_norm(x, mean, var, y, z)


def reduce_as_net(x, y):
    return paddle.reduce_as(x, y)


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
        self.op_name = "pd_op.sum_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sum_net1
        self.enable_cinn = False
        self.tol = 1e-6

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_backward_enabled(True)
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
            core._set_prim_backward_enabled(False)
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


class TestPrimTwoWithGrad(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.add_grad"
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
            core._set_prim_backward_enabled(True)
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
            ops = [
                op.name()
                for op in fn.get_concrete_program(x, y)[-1]
                .program.backward_program.global_block()
                .ops
            ]
            assert self.op_name not in ops
            core._set_prim_backward_enabled(False)
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


class TestPrimBaseOneGradTwoInputs(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "reduce_as_grad"
        self.dtype = "float32"
        self.y_shape = [200, 40]
        self.init_y_shape = [None, 200]
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = reduce_as_net
        self.enable_cinn = False
        self.tol = 1e-5
        self.y_without_grad = True

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_backward_enabled(True)
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
        if self.y_without_grad:
            grad = x.gradient()
        else:
            grad = y.gradient()
        if flag == "prim":
            ops = [
                op.name()
                for op in fn.get_concrete_program(x, y)[-1]
                .program.backward_program.global_block()
                .ops
            ]
            assert self.op_name not in ops
            core._set_prim_backward_enabled(False)
        return res, [grad]

    def test_prim_all_dynamic(self):
        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.tol, atol=self.tol)


class TestPrimThreeWithGrad(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.batch_norm_grad"
        self.dtype = "float32"
        self.x_shape = [30, 40, 50, 60]
        self.init_x_shape = [None, None, None, 60]
        self.y_shape = [40]
        self.init_y_shape = [None]
        self.z_shape = [40]
        self.init_z_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.z = np.random.random(self.z_shape).astype(self.dtype)
        self.net = batch_norm_net1
        self.enable_cinn = False
        self.tol = 1e-5

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_backward_enabled(True)
        x = paddle.to_tensor(self.x, stop_gradient=False)
        y = paddle.to_tensor(self.y, stop_gradient=False)
        z = paddle.to_tensor(self.z, stop_gradient=False)
        if flag == "prim":
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    InputSpec(shape=self.init_x_shape, dtype='float32'),
                    InputSpec(shape=self.init_y_shape, dtype='float32'),
                    InputSpec(shape=self.init_z_shape, dtype='float32'),
                ],
            )
            fn.train()
        else:
            fn = self.net
        res = fn(x, y, z)
        res.backward()
        x_grad = x.gradient()
        y_grad = y.gradient()
        z_grad = z.gradient()
        if flag == "prim":
            ops = [
                op.name()
                for op in fn.get_concrete_program(x, y, z)[-1]
                .program.backward_program.global_block()
                .ops
            ]
            assert self.op_name not in ops
            core._set_prim_backward_enabled(False)
        return res, [x_grad, y_grad, z_grad]

    def test_prim_all_dynamic(self):
        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.tol, atol=self.tol)


if __name__ == "__main__":
    unittest.main()
