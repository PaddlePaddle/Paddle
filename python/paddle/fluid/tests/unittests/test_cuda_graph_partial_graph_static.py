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

import paddle
import paddle.nn as nn
import unittest
import numpy as np
from paddle.device.cuda.graphs import wrap_cuda_graph, is_cuda_graph_supported

paddle.enable_static()


class SimpleModel(nn.Layer):

    def __init__(self, in_size, out_size):
        super(SimpleModel, self).__init__()
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


class TestCudaGraphAttr(unittest.TestCase):

    def test_layer(self):
        if not is_cuda_graph_supported():
            return
        model = SimpleModel(10, 20)
        cuda_graph_model = wrap_cuda_graph(model)
        x = paddle.static.data(shape=[3, 10], dtype='float32')
        y = cuda_graph_model(x)
        program = paddle.static.default_main_program()
        block = program.global_block()
        for op in block.ops:
            assert op._cuda_graph_attr is not None
            print(op._cuda_graph_attr)


if __name__ == "__main__":
    unittest.main()
