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

import sys
import unittest
from os.path import dirname

sys.path.append(dirname(dirname(__file__)))

import numpy as np
import utils

import paddle
from paddle.static import InputSpec


class CINNSliceConcatGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        tmp = x[:, :, :4096]
        tmp1 = tmp.reshape([-1, 1, 32, 128])
        return paddle.concat([tmp1, y], axis=1)


class TestSliceConcat(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        # [batchsize, seqlen, hidden_dim]
        self.x = paddle.uniform(
            [1, 1, 12288], dtype="float16", min=-0.5, max=0.5
        )
        self.x.stop_gradient = True
        # [batchsize, seqlen, num_head, hidden_dim]
        self.y = paddle.uniform(
            [1, 17, 32, 128], dtype="float16", min=-0.5, max=0.5
        )
        self.y.stop_gradient = True

    def train(self, use_cinn):
        net = CINNSliceConcatGraphNet()
        net.eval()
        input_spec = [
            InputSpec([None, 1, 12288], 'float16', 'x'),
            InputSpec([None, None, 32, 128], 'float16', 'y'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        out = net(self.x, self.y)
        return out

    def test_train(self):
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)

        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
