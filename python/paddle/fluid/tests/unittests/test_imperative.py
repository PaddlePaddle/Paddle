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
        return [fluid.layers.elementwise_mul(x, x)]


class TestImperative(unittest.TestCase):
    def test_layer(self):
        cl = core.Layer()
        cl.forward([])
        l = fluid.imperative.PyLayer()
        l.forward([])

    def test_imperative_trace(self):
        with fluid.imperative.guard():
            self.assertTrue(fluid.imperative.enabled())
            x = fluid.layers.data(name='abc', shape=[3, 4], dtype='float32')
            for _ in xrange(2):
                x = fluid.layers.relu(x)
                x = fluid.layers.elementwise_mul(x, x)
                self.assertIsNotNone(x)

    def test_layer_in_out(self):
        l = MyLayer()
        x = l(np.ones([1], np.float32))[0]
        self.assertIsNotNone(x)
        sys.stderr.write("%s output: %s\n" % (x, x.numpy(scope=l._scope)))


if __name__ == '__main__':
    unittest.main()
