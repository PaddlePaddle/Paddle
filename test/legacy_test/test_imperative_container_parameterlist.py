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
from paddle import _legacy_C_ops, base


class MyLayer(paddle.nn.Layer):
    def __init__(self, num_stacked_param, use_base_api):
        super().__init__()
        # create ParameterList with iterable Parameters
        self.params = self.paddle_imperative_ParameterList(num_stacked_param)

    def paddle_imperative_ParameterList(self, num_stacked_param):
        return paddle.nn.ParameterList(
            [
                paddle.create_parameter(shape=[2, 2], dtype='float32')
                for _ in range(num_stacked_param)
            ]
        )

    def forward(self, x):
        for i, p in enumerate(self.params):
            x = _legacy_C_ops.mul(x, p)
        return x


class TestImperativeContainerParameterList(unittest.TestCase):
    def parameter_list(self, use_base_api):
        data_np = np.random.uniform(-1, 1, [5, 2]).astype('float32')
        with base.dygraph.guard():
            x = paddle.to_tensor(data_np)
            num_stacked_param = 4
            model = MyLayer(num_stacked_param, use_base_api)
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

    def test_parameter_list(self):
        self.parameter_list(False)


class TestParameterListAssignment(unittest.TestCase):
    def test_assign_Tensor(self):
        param_list = paddle.nn.ParameterList(
            [
                paddle.create_parameter(shape=[2, 2], dtype='float32'),
                paddle.create_parameter(shape=[2, 2], dtype='float32'),
            ]
        )
        assert isinstance(param_list[0], paddle.base.framework.EagerParamBase)
        assert isinstance(param_list[1], paddle.base.framework.EagerParamBase)

        new_param1 = paddle.randn([2, 3])
        param_list[0] = new_param1
        assert isinstance(param_list[0], paddle.base.framework.EagerParamBase)

        new_param2 = paddle.randn([2, 4])
        param_list[1] = new_param2
        assert isinstance(param_list[1], paddle.base.framework.EagerParamBase)

        np.testing.assert_allclose(new_param1.numpy(), param_list[0].numpy())
        np.testing.assert_allclose(new_param2.numpy(), param_list[1].numpy())

    def test_assign_Parameter(self):
        param_list = paddle.nn.ParameterList(
            [
                paddle.create_parameter(shape=[2, 3], dtype='float32'),
                paddle.create_parameter(shape=[2, 4], dtype='float32'),
            ]
        )
        assert isinstance(param_list[0], paddle.base.framework.EagerParamBase)
        assert isinstance(param_list[1], paddle.base.framework.EagerParamBase)

        new_param1 = paddle.create_parameter([2, 5], dtype='float32')
        param_list[0] = new_param1
        assert isinstance(param_list[0], paddle.base.framework.EagerParamBase)

        new_param2 = paddle.create_parameter([2, 6], dtype='float64')
        param_list[1] = new_param2
        assert isinstance(param_list[1], paddle.base.framework.EagerParamBase)

        np.testing.assert_allclose(new_param1.numpy(), param_list[0].numpy())
        np.testing.assert_allclose(new_param2.numpy(), param_list[1].numpy())

    def test_assign_wrong_type(self):
        param_list = paddle.nn.ParameterList(
            [
                paddle.create_parameter(shape=[2, 2], dtype='float32'),
                paddle.create_parameter(shape=[2, 2], dtype='float32'),
            ]
        )
        with self.assertRaises(TypeError):
            param_list[0] = 1


if __name__ == '__main__':
    unittest.main()
