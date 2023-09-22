#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np

import paddle
from paddle import base

paddle.enable_static()

np.random.seed(123)
os.environ["CPU_NUM"] = "1"
base.core._set_eager_deletion_mode(0.0, 1.0, True)


class RecurrentNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.cell = paddle.nn.SimpleRNNCell(16, 32)
        self.rnn = paddle.nn.RNN(self.cell)

    def forward(self, inputs, prev_h):
        outputs, final_states = self.rnn(inputs, prev_h)
        return outputs, final_states


class TestDy2StRecurrentOpBackward(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        paddle.seed(100)

    def tearDown(self):
        paddle.enable_static()

    def test_recurrent_backward(self):
        net = RecurrentNet()
        inputs = paddle.rand((4, 23, 16))
        inputs.stop_gradient = False
        prev_h = paddle.randn((4, 32))
        prev_h.stop_gradient = False

        outputs, final_states = net(inputs, prev_h)
        outputs.backward()
        dy_grad = inputs.gradient()
        inputs.clear_gradient()

        net = paddle.jit.to_static(net)
        outputs, final_states = net(inputs, prev_h)
        outputs.backward()
        st_grad = inputs.gradient()
        np.testing.assert_allclose(dy_grad, st_grad)


if __name__ == '__main__':
    unittest.main()
