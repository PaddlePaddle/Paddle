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


class L1(fluid.Layer):
    def __init__(self):
        super(L1, self).__init__()
        self._param_attr = fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.1))
        self.w1 = self.create_parameter(
            attr=self._param_attr, shape=[2, 2], dtype='float32', is_bias=False)
        self.w2 = self.create_parameter(
            attr=self._param_attr, shape=[2, 2], dtype='float32', is_bias=False)

    def forward(self):
        return self.w1 + self.w2


class L2(fluid.Layer):
    def __init__(self):
        super(L2, self).__init__()
        self.layer1 = L1()
        self.layer2 = L1()

    def forward(self):
        return self.layer1() + self.layer2()


class L3(fluid.Layer):
    def __init__(self):
        super(L3, self).__init__()
        self.layer1 = L2()
        self.layer2 = L2()

    def forward(self):
        return self.layer1() + self.layer2()


class TestBaseLayer(unittest.TestCase):
    def test_one_level(self):
        with fluid.dygraph.guard():
            l = L1()
            ret = l()
            expected_names = ['l1.w1', 'l1.w2']
            idx = 0
            for name, _ in l.named_parameters(prefix='l1'):
                self.assertEqual(name, expected_names[idx])
                idx += 1
            self.assertTrue(np.allclose(ret.numpy(), 0.2 * np.ones([2, 2])))

    def test_three_level(self):
        with fluid.dygraph.guard():
            l = L3()
            expected_names = [
                'l3.layer1.layer1.w1',
                'l3.layer1.layer1.w2',
                'l3.layer1.layer2.w1',
                'l3.layer1.layer2.w2',
                'l3.layer2.layer1.w1',
                'l3.layer2.layer1.w2',
                'l3.layer2.layer2.w1',
                'l3.layer2.layer2.w2',
            ]
            idx = 0
            for name, _ in l.named_parameters(prefix='l3'):
                self.assertEqual(name, expected_names[idx])
                idx += 1
            ret = l()
            self.assertTrue(np.allclose(ret.numpy(), 0.8 * np.ones([2, 2])))


if __name__ == '__main__':
    unittest.main()
