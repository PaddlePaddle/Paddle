# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import numpy as np


class MyLayer(fluid.Layer):
    def __init__(self, layerlist):
        super(MyLayer, self).__init__()
        self.layerlist = layerlist

    def forward(self, x):
        for l in self.layerlist:
            x = l(x)
        return x


class TestImperativeContainerParameterList(unittest.TestCase):
    def test_paramter_list(self):
        data_np = np.random.uniform(-1, 1, [5, 1]).astype('float32')
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(data_np)
            layerlist = fluid.dygraph.LayerList(
                [fluid.dygraph.Linear(2**i, 2**(i + 1)) for i in range(6)])
            size = len(layerlist)

            model = MyLayer(layerlist)
            res1 = model(x)
            self.assertListEqual(res1.shape, [5, 2**size])
            model.layerlist[size - 1] = fluid.dygraph.Linear(2**(size - 1), 5)
            res2 = model(x)
            self.assertListEqual(res2.shape, [5, 5])
            del model.layerlist[size - 1]
            res3 = model(x)
            self.assertListEqual(res3.shape, [5, 2**(size - 1)])
            model.layerlist.append(fluid.dygraph.Linear(2**(size - 1), 3))
            res4 = model(x)
            self.assertListEqual(res4.shape, [5, 3])
            res4.backward()

            model2 = MyLayer(layerlist[:-1])
            res5 = model2(x)
            self.assertListEqual(res5.shape, [5, 2**(size - 1)])
            del model2.layerlist[1:]
            res6 = model2(x)
            self.assertListEqual(res6.shape, [5, 2**(0 + 1)])
            res6.backward()


if __name__ == '__main__':
    unittest.main()
