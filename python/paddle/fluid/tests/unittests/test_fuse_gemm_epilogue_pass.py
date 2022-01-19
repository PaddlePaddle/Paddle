# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.
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
"""Test cases for role makers."""

from __future__ import print_function
import paddle
import os
import unittest
import numpy as np
import paddle.fluid.core as core


def compare(ref, res, atol, rtol):

    ref = np.array(ref).flatten()
    res = np.array(res).flatten()

    tmp_ref = ref.astype(np.float)
    tol = atol + rtol * abs(tmp_ref)

    diff = abs(res - ref)

    indices = np.transpose(np.where(diff > tol))
    if len(indices) == 0:
        return True
    return False


def verify_node_count(graph, node_name, target_count):
    count = 0
    for node in graph.nodes():
        if node.name() == node_name:
            count += 1
    return count == target_count


class MultiFCLayer(paddle.nn.Layer):
    def __init__(self, hidden):
        super(MultiFCLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(hidden, hidden)
        self.linear2 = paddle.nn.Linear(hidden, hidden)

        self.relu1 = paddle.nn.ReLU()
        self.relu2 = paddle.nn.ReLU()
        self.relu3 = paddle.nn.ReLU()

    def forward(self, x, matmul_y, ele_y):
        output = self.linear1(x)
        output = self.relu1(output)
        output = self.linear2(output)

        output1 = paddle.matmul(output, matmul_y)
        output = self.relu2(output)

        output = paddle.matmul(output, matmul_y)
        output = paddle.add(output, ele_y)
        output = self.relu3(output)
        output = paddle.add(output, output1)
        return output


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueFWDFP32(unittest.TestCase):
    def setUp(self):
        self.batch = 64
        self.seqlen = 128
        self.hidden = 768

        paddle.enable_static()

        self.main_prog = paddle.static.Program()
        self.startup_prog = paddle.static.Program()

        with paddle.static.program_guard(self.main_prog, self.startup_prog):
            data = paddle.static.data(
                name="_data",
                shape=[-1, self.seqlen, self.hidden],
                dtype='float32')
            matmul_y = paddle.static.data(
                name="_matmul_y",
                shape=[1, self.hidden, self.hidden],
                dtype='float32')
            ele_y = paddle.static.data(
                name="_ele_y", shape=[self.hidden, ], dtype='float32')

            multi_layer = MultiFCLayer(self.hidden)
            with paddle.static.amp.fp16_guard():
                out = multi_layer(data, matmul_y, ele_y)
                self.loss = paddle.mean(out)

        self.data_arr = np.random.random(
            (self.batch, self.seqlen, self.hidden)).astype("float32")
        self.matmul_y_arr = np.random.random(
            (1, self.hidden, self.hidden)).astype("float32")
        self.ele_y_arr = np.random.random((self.hidden, )).astype("float32")

        self.place = paddle.CUDAPlace(0)
        self.exe = paddle.static.Executor(self.place)
        self.exe.run(self.startup_prog)

        self._pre_test_hooks()

        self.feed = {
            "_data": self.data_arr,
            "_matmul_y": self.matmul_y_arr,
            "_ele_y": self.ele_y_arr
        }
        self.reference = self.exe.run(self.main_prog,
                                      feed=self.feed,
                                      fetch_list=[self.loss.name])

    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    def test_output(self):
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.fuse_gemm_epilogue = True
        program = paddle.static.CompiledProgram(self.main_prog)
        program = program.with_data_parallel(
            loss_name=self.loss.name,
            build_strategy=build_strategy,
            places=paddle.static.cuda_places())

        result = self.exe.run(program,
                              feed=self.feed,
                              fetch_list=[self.loss.name])
        self.assertTrue(
            compare(self.reference, result, self.atol, self.rtol),
            "[{}] outputs are miss-matched.".format(type(self).__name__))

        self.assertTrue(
            verify_node_count(program._graph, "fused_gemm_epilogue", 1))
        self.assertTrue(verify_node_count(program._graph, "matmul_v2", 3))

    def _pre_test_hooks(self):
        self.atol = 1e-4
        self.rtol = 1e-3


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueFWDFP16(TestFuseGemmEpilogueFWDFP32):
    def _pre_test_hooks(self):
        self.atol = 1e-3
        self.rtol = 1e-2

        fp16_var_list = paddle.static.amp.cast_model_to_fp16(self.main_prog)
        paddle.static.amp.cast_parameters_to_fp16(
            self.place, self.main_prog, to_fp16_var_names=fp16_var_list)

        self.data_arr = self.data_arr.astype("float16")
        self.matmul_y_arr = self.matmul_y_arr.astype("float16")
        self.ele_y_arr = self.ele_y_arr.astype("float16")


if __name__ == "__main__":
    unittest.main()
