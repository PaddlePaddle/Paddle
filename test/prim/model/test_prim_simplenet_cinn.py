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
from paddle.base import core
from paddle.nn import BatchNorm

np.random.seed(2023)


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net, build_strategy=build_strategy, full_graph=True
    )


class PrimeNet(paddle.nn.Layer):
    def __init__(self, shape):
        super().__init__()
        self.bn = BatchNorm(shape[-1], data_layout='NHWC', act="relu")

    def forward(self, data, dout):
        y = self.bn(data) * dout
        return y


class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        self.data = None
        self.dout = None
        self.shape = None

    def train(self, use_prim):
        paddle.seed(2022)
        net = PrimeNet(self.shape)
        sgd = paddle.optimizer.SGD(
            learning_rate=1.0, parameters=net.parameters()
        )
        core._set_prim_all_enabled(use_prim)

        net = paddle.amp.decorate(models=net, level='O2')
        if use_prim:
            net = apply_to_static(net, use_prim)
        res = []
        with paddle.amp.auto_cast(level='O2'):
            for _ in range(10):
                out = net(self.data, self.dout)
                loss = paddle.mean(out)
                loss.backward()
                sgd.step()
                sgd.clear_grad()
                res.append(loss.numpy())
            self.check_prim(net, use_prim)

        return res

    def check_prim(self, net, use_prim):
        if not use_prim:
            return
        fwd_ops = [
            op.type
            for op in net.forward.get_concrete_program(self.data, self.dout)[1]
            .train_program.block(0)
            .ops
        ]

        # Ensure that batch_norm is splitted into small ops
        self.assertTrue('batch_norm' not in fwd_ops)

    def test_cinn_prim(self):
        if paddle.device.get_device() == "cpu":
            return
        self.shape = (16, 112, 112, 64)
        self.data = paddle.to_tensor(
            np.random.random(self.shape).astype("float16")
        )
        self.data.stop_gradient = False
        self.dout = paddle.to_tensor(
            np.random.random(self.shape).astype("float16")
        )

        dy2st_res = self.train(use_prim=False)
        prim_res = self.train(use_prim=True)

        for i in range(len(dy2st_res)):
            np.testing.assert_allclose(
                prim_res[i], dy2st_res[i], rtol=1e-3, atol=1e-3
            )


if __name__ == '__main__':
    unittest.main()
