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
from paddle import nn
from paddle.device.cuda.cuda_graphed_layer import CUDAGraphedLayer

seed = 102


class Model(nn.Layer):
    def __init__(self, in_size, out_size, dropout=0):
        paddle.seed(seed)
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


class DropoutModel(nn.Layer):
    def __init__(self, in_size, out_size, dropout=0.5):
        paddle.seed(seed)
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.dropout_1 = paddle.nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = paddle.nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        return x


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or float(paddle.version.cuda()) < 11.0,
    "only support cuda >= 11.0",
)
class TestSimpleModel(unittest.TestCase):
    def train(self, model):
        paddle.seed(seed)

        ans = []
        for _ in range(10):
            x = paddle.randn([3, 10], dtype='float32')
            x.stop_gradient = False
            loss = model(x).mean()
            loss.backward()
            ans.append(x.grad.numpy())

        return np.array(ans)

    def test_layer(self):
        model = Model(10, 20)
        cuda_graphed_model = CUDAGraphedLayer(Model(10, 20))

        dropout_model = DropoutModel(10, 20)
        cuda_graphed_dropout_model = CUDAGraphedLayer(DropoutModel(10, 20))

        np.testing.assert_array_equal(
            self.train(model), self.train(cuda_graphed_model)
        )
        np.testing.assert_array_equal(
            self.train(dropout_model), self.train(cuda_graphed_dropout_model)
        )


if __name__ == "__main__":
    unittest.main()
