#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

from dygraph_to_static_utils import (
    Dy2StTestBase,
)

import paddle
from paddle.jit.api import to_static

SEED = 102
random.seed(SEED)


class IRSelectedRowsTestNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.embedding = paddle.nn.Embedding(128, 3, sparse=False)

        w0 = paddle.rand([128, 3])
        self.embedding.weight.set_value(w0)

        self.linear = paddle.nn.Linear(
            in_features=3,
            out_features=3,
            weight_attr=paddle.ParamAttr(need_clip=True),
            bias_attr=paddle.ParamAttr(need_clip=False),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x


def forward(net, x):
    loss_data = []
    for _ in range(10):
        out = net(x)
        loss = paddle.mean(out)
        loss_data.append(loss.numpy())
    return loss_data


def forward_dygraph():
    paddle.seed(100)
    net = IRSelectedRowsTestNet()
    x = paddle.randint(low=0, high=128, shape=[64], dtype="int64")

    return forward(net, x)


def forward_static():
    paddle.seed(100)
    net = IRSelectedRowsTestNet()
    x = paddle.randint(low=0, high=128, shape=[64], dtype="int64")

    return to_static(forward, full_graph=True)(net, x)


class TestSimnet(Dy2StTestBase):
    def test_dygraph_static_same_loss(self):
        dygraph_value = forward_dygraph()
        static_value = forward_static()

        self.assertEqual(len(dygraph_value), len(static_value))
        for i in range(len(dygraph_value)):
            self.assertAlmostEqual(dygraph_value[i], static_value[i].numpy())


if __name__ == '__main__':
    unittest.main()
