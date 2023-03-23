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

import unittest

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid import core


class MyModel(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = paddle.nn.Linear(input_size, hidden_size)
        self.linear2 = paddle.nn.Linear(hidden_size, hidden_size)
        self.linear3 = paddle.nn.Linear(hidden_size, 1)
        register_buffer_in_temp = paddle.randn([4, 6])
        self.register_buffer('register_buffer_in', register_buffer_in_temp)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        if (
            paddle.rand(
                [
                    1,
                ]
            )
            > 0.5
        ):
            x = self.linear2(x)
            x = F.relu(x)
        x = self.linear3(x)

        return x


class TestDtypeConvert(unittest.TestCase):
    def setUp(self):
        self.batch_size, self.input_size, self.hidden_size = 128, 128, 256
        self.x_data = np.random.randn(self.batch_size, self.input_size)
        self.y_data = np.random.randn(self.batch_size, 1)

    def test_func_origin(self):
        x = paddle.to_tensor(self.x_data, dtype=paddle.float32)
        y = paddle.to_tensor(self.y_data, dtype=paddle.float32)
        model = MyModel(self.input_size, self.hidden_size)
        loss_fn = paddle.nn.MSELoss(reduction='mean')
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=model.parameters()
        )
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

    def test_func_float(self):
        x = paddle.to_tensor(self.x_data, dtype=paddle.float32)
        y = paddle.to_tensor(self.y_data, dtype=paddle.float32)
        model = MyModel(self.input_size, self.hidden_size)
        model.float()
        loss_fn = paddle.nn.MSELoss(reduction='mean')
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=model.parameters()
        )
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "Require compiled with CUDA."
    )
    def test_func_half(self):
        x = paddle.to_tensor(self.x_data, dtype=paddle.float16)
        y = paddle.to_tensor(self.y_data, dtype=paddle.float16)
        model = MyModel(self.input_size, self.hidden_size)
        model.half()
        loss_fn = paddle.nn.MSELoss(reduction='mean')
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=model.parameters()
        )
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "Require compiled with CUDA."
    )
    def test_func_bfloat16(self):
        x = paddle.to_tensor(self.x_data, dtype=paddle.bfloat16)
        y = paddle.to_tensor(self.y_data, dtype=paddle.bfloat16)
        model = MyModel(self.input_size, self.hidden_size)
        model.bfloat16()

    def batchnorm_half(self, excluded_layers=None):

        x = paddle.to_tensor(self.x_data, dtype=paddle.float32)
        y = paddle.to_tensor(self.y_data, dtype=paddle.float32)

        model = paddle.nn.Sequential(paddle.nn.BatchNorm(self.input_size))

        model.half(excluded_layers=excluded_layers)
        loss_fn = paddle.nn.MSELoss(reduction='mean')
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=model.parameters()
        )
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

    def test_excluded_layers(self):
        self.batchnorm_half(excluded_layers=[nn.BatchNorm])
        self.batchnorm_half(excluded_layers=nn.BatchNorm)
        self.assertRaises(ValueError, self.batchnorm_half)


if __name__ == '__main__':
    unittest.main()
