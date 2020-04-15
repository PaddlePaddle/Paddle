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
import collections
import paddle


class MyLayer(fluid.Layer):
    def __init__(self, layerdict):
        super(MyLayer, self).__init__()
        self.layerdict = layerdict

    def forward(self, x, key):
        return self.layerdict[key](x)


class TestImperativeContainer(unittest.TestCase):
    def fluid_dygraph_linear(self, in_dim, out_dim):
        return fluid.dygraph.Linear(in_dim, out_dim)

    def layer_dict(self):
        data_np = np.random.uniform(-1, 1, [3, 5]).astype('float32')
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(data_np)
            linear1 = self.fluid_dygraph_linear(5, 10)
            linear2 = self.fluid_dygraph_linear(5, 5)

            # use tuple to initialize layerdict
            model1 = MyLayer(fluid.dygraph.LayerDict((('linear1', linear1), )))

            # use dict to initialize layerdict
            model2 = MyLayer(
                fluid.dygraph.LayerDict({
                    'linear1': linear1,
                    'linear2': linear2
                }))

            # use orderdict to initialize layerdict
            orderdict_tmp = collections.OrderedDict()
            orderdict_tmp['linear1'] = linear1
            orderdict_tmp['linear2'] = linear2
            layer_dict_tmp = fluid.dygraph.LayerDict(orderdict_tmp)
            model3 = MyLayer(layer_dict_tmp)

            # use layerdict to initialize layerdict
            model4 = MyLayer(fluid.dygraph.LayerDict(layer_dict_tmp))

            model1_linear1 = model1(x, 'linear1')
            model1_linear1.backward()

            model2_linear1 = model2(x, 'linear1')
            model2_linear2 = model2(x, 'linear2')

            model3_linear1 = model3(x, 'linear1')
            model3_linear2 = model3(x, 'linear2')

            model4_linear1 = model4(x, 'linear1')
            model4_linear2 = model4(x, 'linear2')

            self.assertListEqual(model1_linear1.shape, model2_linear1.shape)
            self.assertTrue(
                np.array_equal(model1_linear1.numpy(), model2_linear1.numpy()))
            self.assertListEqual(model1_linear1.shape, model3_linear1.shape)
            self.assertTrue(
                np.array_equal(model1_linear1.numpy(), model3_linear1.numpy()))
            self.assertListEqual(model1_linear1.shape, model4_linear1.shape)
            self.assertTrue(
                np.array_equal(model1_linear1.numpy(), model4_linear1.numpy()))

            self.assertListEqual(model2_linear2.shape, model3_linear2.shape)
            self.assertTrue(
                np.array_equal(model2_linear2.numpy(), model3_linear2.numpy()))
            self.assertListEqual(model2_linear2.shape, model4_linear2.shape)
            self.assertTrue(
                np.array_equal(model2_linear2.numpy(), model4_linear2.numpy()))

    def test_layer_dict(self):
        self.layer_dict()

    def test_layer_dict_basic(self):
        linear1 = self.fluid_dygraph_linear(5, 10)
        linear2 = self.fluid_dygraph_linear(5, 5)
        layer_dict_tmp = fluid.dygraph.LayerDict({
            'linear1': linear1,
            'linear2': linear1
        })
        linear3 = self.fluid_dygraph_linear(5, 15)
        layer_dict_tmp['linear3'] = linear3

        # test __iter__
        for i in (layer_dict_tmp):
            pass

        self.assertTrue('linear3' in layer_dict_tmp)
        self.assertTrue(len(layer_dict_tmp) == 3)
        del layer_dict_tmp['linear3']
        self.assertTrue('linear3' not in layer_dict_tmp)
        self.assertTrue(len(layer_dict_tmp) == 2)

        layer_dict_tmp.pop('linear2')
        self.assertTrue('linear2' not in layer_dict_tmp)
        self.assertTrue(len(layer_dict_tmp) == 1)

        layer_dict_tmp.clear()
        self.assertTrue(len(layer_dict_tmp) == 0)


if __name__ == '__main__':
    unittest.main()
