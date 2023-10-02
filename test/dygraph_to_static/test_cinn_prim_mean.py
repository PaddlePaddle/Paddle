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
from dygraph_to_static_util import ast_only_test, dy2static_unittest

import paddle
from paddle import tensor
from paddle.base import core

TOLERANCE = {
    "float16": {"rtol": 1e-3, "atol": 1e-3},
    "float32": {"rtol": 1e-6, "atol": 1e-6},
    "float64": {"rtol": 1e-15, "atol": 1e-15},
}

keepdim_conds = [True, False]
axes_condis = [-1, 0, 1]


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)


def generate_data(shape, dtype="float32"):
    np_data = np.random.random(shape).astype(dtype)
    return np_data


class PrimeNet(
    paddle.nn.Layer,
):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x):
        # y = self.fc(x)
        out = tensor.mean(x)
        return out


@dy2static_unittest
class TestPrimForward(unittest.TestCase):
    """
    This case only tests prim_forward + to_static + cinn. Thus we need to
    set this flag as False to avoid prim_backward.
    core.set_prim_backward(False)
    """

    def setUp(self):
        paddle.seed(2022)
        self.shapes = [[2, 4], [64, 16, 4]]
        self.dtypes = ["float16", "float32", "float64"]

    def train(self, use_prim, data):
        for keep_dim in keepdim_conds:
            for axis in axes_condis:
                return self._train(use_prim, data, axis, keep_dim)

    def _train(self, use_prim, data, axis, keep_dim):
        paddle.seed(2022)
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )
        core._set_prim_forward_enabled(use_prim)
        if use_prim:
            net = apply_to_static(net, use_prim)

        res = []
        self.x = data
        for _ in range(10):
            out = net(data)
            loss = paddle.mean(out, axis, keep_dim)
            loss.backward()
            sgd.step()
            sgd.clear_grad()

            res.append(out.numpy())

        self.check_prim(net, use_prim)

        return res

    def check_prim(self, net, use_prim):
        if not use_prim:
            return
        # Please use PartialProgramLayer(second output parameter of get_concrete_program) rather than
        # main_program here, as main_program is original program before to_prim.
        fwd_ops = [
            op.type
            for op in net.forward.get_concrete_program(self.x)[1]
            .train_program.block(0)
            .ops
        ]
        # Ensure that reduce_mean is splitted into small ops
        self.assertTrue('reduce_mean' not in fwd_ops)

    @ast_only_test
    def test_cinn_prim_forward(self):
        for shape in self.shapes:
            for dtype in self.dtypes:
                # mean-kernel on cpu not support float16
                if paddle.device.get_device() == "cpu" and dtype == "float16":
                    print("need pass this case")
                    continue
                data = generate_data(shape, dtype)
                data_t = paddle.to_tensor(data)
                data_t.stop_gradient = False
                dy_res = self.train(use_prim=False, data=data_t)
                cinn_res = self.train(use_prim=True, data=data_t)

                np.testing.assert_allclose(
                    cinn_res,
                    dy_res,
                    rtol=TOLERANCE[dtype]['rtol'],
                    atol=TOLERANCE[dtype]['atol'],
                )


@dy2static_unittest
class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        paddle.seed(2022)
        self.shapes = [[2, 4], [64, 16, 4]]
        self.dtypes = ["float16", "float32", "float64"]

    def train(self, use_prim, data):
        for keep_dim in keepdim_conds:
            for axis in axes_condis:
                return self._train(use_prim, data, axis, keep_dim)

    def _train(self, use_prim, data, axis, keep_dim):
        paddle.seed(2022)
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )
        core._set_prim_all_enabled(use_prim)
        if use_prim:
            net = apply_to_static(net, use_prim)

        res = []
        self.x = data
        for _ in range(10):
            out = net(data)
            loss = paddle.mean(out, axis, keep_dim)
            loss.backward()
            sgd.step()
            sgd.clear_grad()

            res.append(out.numpy())

        self.check_prim(net, use_prim)

        return res

    def check_prim(self, net, use_prim):
        if not use_prim:
            return
        fwd_ops = [
            op.type
            for op in net.forward.get_concrete_program(self.x)[1]
            .train_program.block(0)
            .ops
        ]
        # Ensure that reduce_mean is splitted into small ops
        self.assertTrue('reduce_mean' not in fwd_ops)

    @ast_only_test
    def test_cinn_prim(self):
        for shape in self.shapes:
            for dtype in self.dtypes:
                # mean-kernel on cpu not support float16
                if paddle.device.get_device() == "cpu" and dtype == "float16":
                    print("need pass this case")
                    continue
                data = generate_data(shape, dtype)
                data_t = paddle.to_tensor(data)
                data_t.stop_gradient = False
                dy_res = self.train(use_prim=False, data=data_t)
                cinn_res = self.train(use_prim=True, data=data_t)

                np.testing.assert_allclose(
                    cinn_res,
                    dy_res,
                    rtol=TOLERANCE[dtype]['rtol'],
                    atol=TOLERANCE[dtype]['atol'],
                )


if __name__ == '__main__':
    unittest.main()
