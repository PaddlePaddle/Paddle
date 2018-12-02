# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import numpy as np

import paddle.fluid as fluid
from paddle.fluid import core


class MyLayer(fluid.imperative.PyLayer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def forward(self, inputs):
        x = fluid.layers.relu(inputs[0])
        self._x_for_debug = x
        return [fluid.layers.elementwise_mul(x, x)]


class TestImperative(unittest.TestCase):
    def test_layer(self):
        with fluid.imperative.guard():
            cl = core.Layer()
            cl.forward([])
            l = fluid.imperative.PyLayer()
            l.forward([])

    def test_layer_in_out(self):
        with fluid.imperative.guard():
            l = MyLayer()
            x = l(np.array([1.0, 2.0, -1.0], dtype=np.float32))[0]
            self.assertIsNotNone(x)
            sys.stderr.write("%s output: %s\n" % (x, x.numpy(scope=l._scope)))
            x.backward(l._scope)
            sys.stderr.write("grad %s\n" % l._x_for_debug.grad())


if __name__ == '__main__':
    unittest.main()
