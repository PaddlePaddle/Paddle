# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest
import paddle
import numpy as np
import random
from paddle.fluid.dygraph.container import BufferList
from numpy.testing import assert_array_equal
from paddle import fluid


class SimpleNet(paddle.nn.Layer):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear1 = paddle.nn.Linear(10, 3)
        self.linear2 = paddle.nn.Linear(3, 1)
        self.hidden = BufferList([
            paddle.to_tensor([1.0]),
            paddle.to_tensor([2.0]),
        ])
        #self.hidden  = paddle.to_tensor([1.0])

    def forward(self, x):
        out1 = self.linear1(x)
        out2 = self.linear2(out1) + self.hidden[0]
        self.hidden.assign([paddle.to_tensor([3.0]), paddle.to_tensor([4.0])])
        return [out1, out2]


def run_graph(inp, to_static):
    paddle.seed(2021)
    np.random.seed(2021)
    random.seed(2021)
    net = SimpleNet()
    if to_static:
        net = paddle.jit.to_static(net)
    loss = net(inp)
    return loss


class TestBufferList(unittest.TestCase):

    def test_bufferlist(self):
        inp = paddle.rand((10, ))
        assert_array_equal(
            run_graph(inp, True)[0].numpy(),
            run_graph(inp, False)[0].numpy(),
            "Not Equal in dygraph and static graph", True)


if __name__ == '__main__':
    with fluid.framework._test_eager_guard():
        unittest.main()
