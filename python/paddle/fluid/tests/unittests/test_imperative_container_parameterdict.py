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
    def __init__(self, parameterdict):
        self.parameterdict = parameterdict

    def forward(self, x, key):
        tmp = self._helper.create_variable_for_type_inference('float32')
        self._helper.append_op(
            type="mul",
            inputs={"X": x,
                    "Y": self.parameterdict[key]},
            outputs={"Out": tmp},
            attrs={"x_num_col_dims": 1,
                   "y_num_col_dims": 1})
        x = tmp
        return x


class TestImperativeContainerParameterDict(unittest.TestCase):
    def parameter_dict(self):
        data_np = np.random.uniform(-1, 1, [3, 5]).astype('float32')
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(data_np)
            param1 = fluid.layers.create_parameter(
                shape=[5, 10], dtype='float32')
            param2 = fluid.layers.create_parameter(
                shape=[5, 5], dtype='float32')

            # use tuple to initialize parameterdict
            model1 = MyLayer(
                fluid.dygraph.ParameterDict(tuple('param1', param1)))

            # use dict to initialize parameterdict
            model2 = MyLayer(
                fluid.dygraph.ParameterDict({
                    'param1': param1,
                    'param2': param2
                }))

            # use orderdict to initialize parameterdict
            orderdict_temp = collections.OrderedDict()
            orderdict_temp['param1'] = param1
            orderdict_temp['param2'] = param2
            model3 = MyLayer(fluid.dygraph.ParameterDict(orderdict_temp))

            # use parameterdict to initialize parameterdict
            model4 = MyLayer(fluid.dygraph.ParameterDict(model3))

            model1_param1 = model1(x, 'param1')
            model1_param1.backward()

            model2_param1 = model2(x, 'param1')
            model2_param2 = model2(x, 'param2')

            model3_param1 = model3(x, 'param1')
            model3_param2 = model3(x, 'param2')

            model4_param1 = model4(x, 'param1')
            model4_param2 = model4(x, 'param2')

            self.assertListEqual(model1_param1.shape, model2_param1.shape)
            self.assertTrue(np.array_equal(model1_param1, model2_param1))
            self.assertListEqual(model1_param1.shape, model3_param1.shape)
            self.assertTrue(np.array_equal(model1_param1, model3_param1))
            self.assertListEqual(model1_param1.shape, model4_param1.shape)
            self.assertTrue(np.array_equal(model1_param1, model4_param1))

            self.assertListEqual(model2_param2.shape, model3_param2.shape)
            self.assertTrue(np.array_equal(model2_param2, model3_param2))
            self.assertListEqual(model2_param2.shape, model4_param2.shape)
            self.assertTrue(np.array_equal(model2_param2, model4_param2))

    def test_parameter_dict():
        self.parameter_dict()


if __name__ == '__main__':
    unittest.main()
