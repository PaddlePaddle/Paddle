#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
)

import paddle

np.random.seed(1)


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = paddle.nn.Linear(1, 1)

    def forward(self, x):
        """forward with duplicate outputs."""
        x = self._linear(x)
        return x, x


class DuplicateOutputInPaddleLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        # In GRUCell, the output is a tuple (h, h)
        self.layer = paddle.nn.GRUCell(10, 20)

    def forward(self, x):
        x = self.layer(x)
        return x


class TestDuplicateOutput(Dy2StTestBase):
    def _run_static(self):
        net = paddle.jit.to_static(SimpleNet())
        x = paddle.to_tensor([1.0])
        param = net.parameters()
        param[0].clear_grad()

        loss0, loss1 = net(x)
        loss0.backward()

        self.assertEqual(param[0].grad.numpy(), 1.0)

    def test_ast_to_func(self):
        self._run_static()


class TestDuplicateOutputInPaddleLayer(Dy2StTestBase):
    def check_dygraph_and_static_result(self, net, x):
        static_net = paddle.jit.to_static(net)
        dy_out = net(x)
        st_out = static_net(x)
        np.testing.assert_allclose(dy_out, st_out)

    def test_ast_to_func(self):
        net = DuplicateOutputInPaddleLayer()
        x = paddle.randn([10, 10])
        self.check_dygraph_and_static_result(net, x)


if __name__ == '__main__':
    unittest.main()
