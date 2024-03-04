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

import random
import unittest

import numpy as np
from numpy.testing import assert_array_equal

import paddle
from paddle.jit.sot import symbolic_translate
from paddle.jit.sot.utils.utils import execute_time
from paddle.vision import resnet50


def resnet_call(net: paddle.nn.Layer, x: paddle.Tensor):
    return net(x)


def run_dygraph_optimizer(inp):
    """dygraph train + SGD optimizer"""
    paddle.seed(2021)
    np.random.seed(2021)
    random.seed(2021)
    net = resnet50()
    optimizer = paddle.optimizer.SGD(
        learning_rate=0.03, parameters=net.parameters()
    )
    for i in range(5):
        optimizer.clear_grad()
        loss = execute_time(net)(inp)
        loss.backward()
        optimizer.step()
    return loss


def run_symbolic_optimizer(inp):
    """dygraph train + SGD optimizer"""
    paddle.seed(2021)
    np.random.seed(2021)
    random.seed(2021)
    net = resnet50()
    net_wrapper = symbolic_translate(resnet_call)
    optimizer = paddle.optimizer.SGD(
        learning_rate=0.03, parameters=net.parameters()
    )
    for i in range(5):
        optimizer.clear_grad()
        loss = execute_time(net_wrapper)(net, inp)
        loss.backward()
        optimizer.step()
    return loss


class TestBackward(unittest.TestCase):
    def test(self):
        # TODO(xiongkun) add cache to speedup !
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        inp = paddle.rand((3, 3, 255, 255))
        out2 = run_symbolic_optimizer(inp)[0].numpy()
        out1 = run_dygraph_optimizer(inp)[0].numpy()
        assert_array_equal(
            out1, out2, "Not Equal in dygraph and static graph", True
        )


if __name__ == "__main__":
    unittest.main()
