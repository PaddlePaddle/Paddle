# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.device.cuda.graphs import is_cuda_graph_supported, wrap_cuda_graph


class SimpleModel(nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.dropout_1 = paddle.nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.dropout_2 = paddle.nn.Dropout(0.5)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.gelu(x)
        return x


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or float(paddle.version.cuda()) < 11.0,
    "only support cuda >= 11.0",
)
class TestSimpleModel(unittest.TestCase):
    def setUp(self):
        paddle.set_flags({'FLAGS_eager_delete_tensor_gb': 0.0})

    def run_base(self, func, use_cuda_graph, memory_pool="default", seed=10):
        paddle.seed(seed)
        is_layer = isinstance(func, paddle.nn.Layer)
        if use_cuda_graph:
            func = wrap_cuda_graph(func, memory_pool=memory_pool)

        for _ in range(10):
            x = paddle.randn([3, 10], dtype='float32')
            x.stop_gradient = False
            y = x * x + 100
            loss = func(y).mean()
            loss.backward()
            if is_layer:
                func.clear_gradients()

        return func, x.grad.numpy()

    def check(self, func):
        if not is_cuda_graph_supported():
            return

        _, value1 = self.run_base(func, False)
        layer, value2 = self.run_base(func, True, "default")
        _, value3 = self.run_base(func, True, "new")
        _, value4 = self.run_base(func, True, layer)
        np.testing.assert_array_equal(value1, value2)
        np.testing.assert_array_equal(value1, value3)
        np.testing.assert_array_equal(value1, value4)

    def test_layer(self):
        self.check(SimpleModel(10, 20))


if __name__ == "__main__":
    unittest.main()
