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
from paddle.device.cuda.graphs import (
    cuda_graph_transform,
    is_cuda_graph_supported,
    wrap_cuda_graph,
)

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
    def setUp(self):
        paddle.set_flags({'FLAGS_eager_delete_tensor_gb': 0.0})

    def get_model(self, use_cuda_graph=False):
        x = paddle.static.data(shape=[3, 10], dtype='float32', name='x')

        model_start = SimpleModel(10, 20)
        if use_cuda_graph:
            model_start = wrap_cuda_graph(model_start)

        model_inter = SimpleModel(20, 20)

        model_end = SimpleModel(20, 10)
        if use_cuda_graph:
            model_end = wrap_cuda_graph(model_end, memory_pool='new')

        start_out = model_start(x)
        inter_out = model_inter(start_out)
        end_out = model_end(inter_out)
        loss = paddle.mean(end_out)

        opt = paddle.optimizer.SGD()
        opt.minimize(loss)

        return loss

    def run_with_cuda_graph(self, x_data):
        # run with cuda graph
        paddle.seed(1024)

        main_prog = paddle.static.Program()
        start_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, start_prog):
            loss = self.get_model(use_cuda_graph=True)

        section_programs = cuda_graph_transform(main_prog)
        assert len(section_programs) == 4

        block = main_prog.global_block()
        run_program_op_num = 0
        for op in block.ops:
            if op.type == 'run_program':
                run_program_op_num += 1
        assert run_program_op_num == 4

        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        exe.run(start_prog)

        for i in range(10):
            rst = exe.run(main_prog, feed={'x': x_data}, fetch_list=[loss])

        return rst

    def normal_run(self, x_data):
        # run without cuda graph
        paddle.seed(1024)

        main_prog = paddle.static.Program()
        start_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, start_prog):
            loss = self.get_model()

        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        exe.run(start_prog)

        for i in range(10):
            rst = exe.run(main_prog, feed={'x': x_data}, fetch_list=[loss])

        return rst

    def test_static_mode_cuda_graph(self):
        if not is_cuda_graph_supported():
            return
        x_data = np.random.random((3, 10)).astype('float32')
        cuda_graph_rst = self.run_with_cuda_graph(x_data)
        normal_run_rst = self.normal_run(x_data)
        np.testing.assert_array_equal(cuda_graph_rst, normal_run_rst)


if __name__ == "__main__":
    unittest.main()
