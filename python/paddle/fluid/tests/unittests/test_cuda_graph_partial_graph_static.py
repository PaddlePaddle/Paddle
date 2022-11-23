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
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
from paddle.device.cuda.graphs import wrap_cuda_graph, is_cuda_graph_supported

paddle.enable_static()


class SimpleModel(nn.Layer):
<<<<<<< HEAD

    def __init__(self, in_size, out_size):
        super(SimpleModel, self).__init__()
=======
    def __init__(self, in_size, out_size):
        super().__init__()
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
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


class TestCudaGraphAttrAll(unittest.TestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def test_all_program(self):
        if not is_cuda_graph_supported():
            return
        main_prog = paddle.static.Program()
        start_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            model = SimpleModel(10, 20)
            cuda_graph_model = wrap_cuda_graph(model)
            x = paddle.static.data(shape=[3, 10], dtype='float32', name='x')
            y = cuda_graph_model(x)
            loss = paddle.mean(y)
            opt = paddle.optimizer.SGD()
            opt.minimize(loss)
            block = main_prog.global_block()
            for op in block.ops:
                if op._cuda_graph_attr is None:
                    # the loss and opt are not wrapped
                    assert op.type in [
<<<<<<< HEAD
                        'sgd', 'reduce_mean', 'fill_constant',
                        'reduce_mean_grad'
=======
                        'sgd',
                        'reduce_mean',
                        'fill_constant',
                        'reduce_mean_grad',
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                    ]
                else:
                    assert op._cuda_graph_attr == 'thread_local;0;0'


if __name__ == "__main__":
    unittest.main()
