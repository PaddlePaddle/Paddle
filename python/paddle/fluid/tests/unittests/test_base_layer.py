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
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper


class L1(fluid.imperative.Layer):
    def __init__(self):
        super(L1, self).__init__()
        self._helper = LayerHelper(
            'MyLayer',
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.1)))

        self.w1 = self._helper.create_parameter(
            attr=self._helper.param_attr,
            shape=[2, 2],
            dtype='float32',
            is_bias=False)
        self.w2 = self._helper.create_parameter(
            attr=self._helper.param_attr,
            shape=[2, 2],
            dtype='float32',
            is_bias=False)

    def forward(self):
        return self.w1 + self.w2


class L2(fluid.imperative.Layer):
    def __init__(self):
        super(L2, self).__init__()
        self.layer1 = L1()
        self.layer2 = L1()

    def forward(self):
        return self.layer1() + self.layer2()


class L3(fluid.imperative.Layer):
    def __init__(self):
        super(L3, self).__init__()
        self.layer1 = L2()
        self.layer2 = L2()

    def forward(self):
        return self.layer1() + self.layer2()


class TestBaseLayer(unittest.TestCase):
    def test_one_level(self):
        with fluid.imperative.guard():
            l = L1()
            ret = l()
            self.assertEqual(l.w1.name, "MyLayer_0.w_0")
            self.assertEqual(l.w2.name, "MyLayer_0.w_1")
            self.assertTrue(np.allclose(ret._numpy(), 0.2 * np.ones([2, 2])))

    def test_three_level(self):
        with fluid.imperative.guard():
            l = L3()
            ret = l()
            self.assertTrue(np.allclose(ret._numpy(), 0.8 * np.ones([2, 2])))


if __name__ == '__main__':
    unittest.main()
