# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
)

import paddle

SEED = 2020


class Pool2D(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.pool2d = paddle.nn.AvgPool2D(kernel_size=2, stride=1)

    def forward(self, x):
        # Add func `get_result` for testing arg_name_to_idx in ast transformation.
        def get_result(x):
            return self.pool2d(x)

        pre = get_result(x)
        return pre


class Linear(paddle.nn.Layer):
    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.fc = paddle.nn.Linear(
            input_dim,
            output_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.99)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            ),
        )
        self.act = paddle.nn.ReLU()

    def forward(self, x):
        pre = self.fc(x)
        pre = self.act(pre)
        loss = paddle.mean(pre)
        return pre, loss


class TestPool2D(Dy2StTestBase):
    def setUp(self):
        self.dygraph_class = Pool2D
        self.data = np.random.random((1, 2, 4, 4)).astype('float32')

    def train(self):
        dy_layer = paddle.jit.to_static(self.dygraph_class())
        x = paddle.to_tensor(self.data)
        prediction = dy_layer(x)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        return prediction.numpy()

    def train_static(self):
        with enable_to_static_guard(True):
            return self.train()

    def train_dygraph(self):
        with enable_to_static_guard(False):
            return self.train()

    def test_to_static(self):
        dygraph_res = self.train_dygraph()
        static_res = self.train_static()

        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
        )


class TestLinear(TestPool2D):
    def setUp(self):
        self.dygraph_class = Linear
        self.data = np.random.random((4, 10)).astype('float32')


if __name__ == '__main__':
    unittest.main()
