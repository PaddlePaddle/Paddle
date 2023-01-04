# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
from paddle import _legacy_C_ops


class MyLayer(fluid.Layer):
    def __init__(self, num_stacked_param, use_fluid_api):
        super().__init__()
        # create ParameterList with iterable Parameters
        self.params = self.paddle_imperative_ParameterList(num_stacked_param)

    def paddle_imperative_ParameterList(self, num_stacked_param):
        return paddle.nn.ParameterList(
            [paddle.create_parameter(shape=[2, 2], dtype='float32')]
            * num_stacked_param
        )

    def forward(self, x):
        for i, p in enumerate(self.params):
            x = _legacy_C_ops.mul(x, p)
        return x


class TestImperativeContainerParameterList(unittest.TestCase):
    def paramter_list(self, use_fluid_api):
        data_np = np.random.uniform(-1, 1, [5, 2]).astype('float32')
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(data_np)
            num_stacked_param = 4
            model = MyLayer(num_stacked_param, use_fluid_api)
            self.assertEqual(len(model.params), num_stacked_param)
            res = model(x)
            self.assertListEqual(res.shape, [5, 2])
            loss = paddle.mean(res)
            loss.backward()

            model.params[num_stacked_param - 1] = paddle.create_parameter(
                shape=[2, 3], dtype='float32'
            )
            res = model(x)
            self.assertListEqual(res.shape, [5, 3])
            model.params.append(
                paddle.create_parameter(shape=[3, 4], dtype='float32')
            )
            self.assertEqual(len(model.params), num_stacked_param + 1)
            res = model(x)
            self.assertListEqual(res.shape, [5, 4])
            loss = paddle.mean(res)
            loss.backward()

    def test_paramter_list(self):
        self.paramter_list(False)
        self.paramter_list(True)


if __name__ == '__main__':
    unittest.main()
