# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np

SEED = 2020
np.random.seed(SEED)


class SimpleNet(paddle.nn.Layer):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear1 = paddle.nn.Linear(10, 3)
        self.linear2 = paddle.nn.Linear(3, 1)

    def forward(self, x):
        out1 = self.linear1(x)
        out2 = self.linear2(out1)
        return [out1, out2]  # 梯度为0
        #return [out1]        # 梯度正常
        #return [out2, out1] # 梯度正常


class TestGradientAggregationInDy2Static(unittest.TestCase):

    def test_to_static(self):

        def simplenet_grad(inp, to_static=False):
            net = SimpleNet()
            if to_static: net = paddle.jit.to_static(net)
            loss = net(inp)
            loss[0].backward()
            return net.linear1.weight.grad

        inp = paddle.to_tensor(np.random.randn(10, )).astype("float32")
        np.testing.assert_allclose(simplenet_grad(inp, True).numpy(),
                                   simplenet_grad(inp, False).numpy(),
                                   rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
