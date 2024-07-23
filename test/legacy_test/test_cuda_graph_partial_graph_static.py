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

import paddle
from paddle import nn
from paddle.device.cuda.graphs import is_cuda_graph_supported, wrap_cuda_graph

paddle.enable_static()


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
class TestCudaGraphAttrAll(unittest.TestCase):
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
                if not paddle.framework.use_pir_api():
                    if op._cuda_graph_attr is None:
                        # the loss and opt are not wrapped
                        assert op.type in [
                            'sgd',
                            'reduce_mean',
                            'fill_constant',
                            'reduce_mean_grad',
                        ]
                    else:
                        assert op._cuda_graph_attr == 'thread_local;0;0'


if __name__ == "__main__":
    unittest.main()
