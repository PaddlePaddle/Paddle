#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.nn as nn


class LeNetDygraph(fluid.dygraph.Layer):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2D(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            paddle.nn.MaxPool2D(2, 2),
            nn.Conv2D(6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            paddle.nn.MaxPool2D(2, 2),
        )

    def forward(self, inputs):
        x = self.features(inputs)
        return x


class TestLayerChildren(unittest.TestCase):
    def func_apply_init_weight(self):
        with fluid.dygraph.guard():
            net = LeNetDygraph()
            net.eval()

            net_layers = nn.Sequential(*list(net.children()))
            net_layers.eval()

            x = paddle.rand([2, 1, 28, 28])
            y1 = net(x)
            y2 = net_layers(x)

            np.testing.assert_allclose(y1.numpy(), y2.numpy())
            return y1, y2

    def test_func_apply_init_weight(self):
        paddle.seed(102)
        self.new_y1, self.new_y2 = self.func_apply_init_weight()
        paddle.seed(102)
        self.ori_y1, self.ori_y2 = self.func_apply_init_weight()

        # compare ori dygraph and new egr
        assert np.array_equal(self.ori_y1.numpy(), self.new_y1.numpy())
        assert np.array_equal(self.ori_y2.numpy(), self.new_y2.numpy())


if __name__ == '__main__':
    unittest.main()
