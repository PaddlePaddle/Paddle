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
from dygraph_to_static_util import ast_only_test

import paddle
import paddle.nn.functional as F
from paddle.base import core

TOLERANCE = {
    "float16": {"rtol": 1e-2, "atol": 1e-2},
    "float32": {"rtol": 1e-5, "atol": 1e-5},
    "float64": {"rtol": 1e-13, "atol": 1e-13},
}


def generate_data(dtype="float32"):
    np_data1 = np.random.random([2, 64]).astype(dtype)
    np_data2 = np.random.random([64]).astype(dtype)
    np_data3 = np.random.random([64]).astype(dtype)
    return np_data1, np_data2, np_data3


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(64, 64)

    def forward(self, x, w, b):
        n_shape = x.shape[1:]
        out = F.layer_norm(x, n_shape, w, b)
        return out[0]


class TestPrimForward(unittest.TestCase):
    """
    This case only tests prim_forward + to_static + cinn. Thus we need to
    set this flag as False to avoid prim_backward.
    core.set_prim_backward(False)
    """

    def setUp(self):
        self.x = None
        self.w = None
        self.b = None
        self.dtypes = ["float16", "float32"]

    def train(self, use_prim):
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )
        core._set_prim_forward_enabled(use_prim)
        core._add_skip_comp_ops("sqrt")
        # TODO(Ruting) delete this after modify sqrt
        if use_prim:
            net = apply_to_static(net, use_prim)

        out = net(self.x, self.w, self.b)
        loss = paddle.mean(out)
        loss.backward()
        sgd.step()
        sgd.clear_grad()

        self.check_prim(net, use_prim)

        return out.numpy()

    def check_prim(self, net, use_prim):
        if not use_prim:
            return
        # Please use PartialProgramLayer(second output parameter of get_concrete_program) rather than
        # main_program here, as main_program is original program before to_prim.
        fwd_ops = [
            op.type
            for op in net.forward.get_concrete_program(self.x, self.w, self.b)[
                1
            ]
            .train_program.block(0)
            .ops
        ]
        # Ensure that layer_norm is splitted into small ops
        self.assertTrue('layer_norm' not in fwd_ops)

    @ast_only_test
    def test_cinn_prim_forward(self):
        for dtype in self.dtypes:
            if paddle.device.get_device() == "cpu":
                print("need pass this case")
                continue
            x_n, w_n, b_n = generate_data(dtype)
            self.x = paddle.to_tensor(x_n)
            self.w = paddle.to_tensor(w_n)
            self.b = paddle.to_tensor(b_n)
            self.x.stop_gradient = False
            dy_res = self.train(use_prim=False)
            cinn_res = self.train(use_prim=True)

            np.testing.assert_allclose(
                cinn_res,
                dy_res,
                rtol=TOLERANCE[dtype]['rtol'],
                atol=TOLERANCE[dtype]['atol'],
            )


class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        self.x = None
        self.w = None
        self.b = None
        self.dtypes = ["float16", "float32"]

    def train(self, use_prim):
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )
        core._set_prim_all_enabled(use_prim)
        core._add_skip_comp_ops("sqrt")
        # TODO(Ruting) delete this after modify sqrt
        if use_prim:
            net = apply_to_static(net, use_prim)

        out = net(self.x, self.w, self.b)
        loss = paddle.mean(out)
        loss.backward()
        sgd.step()
        sgd.clear_grad()

        self.check_prim(net, use_prim)

        return out.numpy()

    def check_prim(self, net, use_prim):
        if not use_prim:
            return
        fwd_ops = [
            op.type
            for op in net.forward.get_concrete_program(self.x, self.w, self.b)[
                1
            ]
            .train_program.block(0)
            .ops
        ]
        # Ensure that layer_norm is splitted into small ops
        self.assertTrue('layer_norm' not in fwd_ops)

    @ast_only_test
    def test_cinn_prim(self):
        for dtype in self.dtypes:
            if paddle.device.get_device() == "cpu":
                print("need pass this case")
                continue
            x_n, w_n, b_n = generate_data(dtype)
            self.x = paddle.to_tensor(x_n)
            self.w = paddle.to_tensor(w_n)
            self.b = paddle.to_tensor(b_n)
            self.x.stop_gradient = False
            dy_res = self.train(use_prim=False)
            cinn_res = self.train(use_prim=True)

            np.testing.assert_allclose(
                cinn_res,
                dy_res,
                rtol=TOLERANCE[dtype]['rtol'],
                atol=TOLERANCE[dtype]['atol'],
            )


if __name__ == '__main__':
    unittest.main()
