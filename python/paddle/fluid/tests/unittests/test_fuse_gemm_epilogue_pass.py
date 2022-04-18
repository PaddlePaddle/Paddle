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
    def __init__(self, hidden, Activation):
        super(MultiFCLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(hidden, hidden)
        self.linear2 = paddle.nn.Linear(hidden, hidden)
        self.linear3 = paddle.nn.Linear(hidden, hidden)

        self.relu1 = Activation()
        self.relu2 = Activation()
        self.relu3 = Activation()

    def forward(self, x, matmul_y, ele_y):
        output = self.linear1(x)
        output = self.relu1(output)
        output = self.linear2(output)

        output1 = paddle.matmul(output, matmul_y)
        output = self.linear3(output)
        output = self.relu2(output)

        output = paddle.matmul(output, matmul_y)
        output = paddle.add(output, ele_y)
        output = self.relu3(output)
        output = paddle.add(output, output1)
        return output


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueFWDBase(unittest.TestCase):
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

            multi_layer = MultiFCLayer(self.hidden, self._get_act_type()[0])
            with paddle.static.amp.fp16_guard():
                out = multi_layer(data, matmul_y, ele_y)
                self.loss = paddle.mean(out)

        self.data_arr = np.random.random(
            (self.batch, self.seqlen, self.hidden)).astype("float32") - 0.5
        self.matmul_y_arr = np.random.random(
            (1, self.hidden, self.hidden)).astype("float32") - 0.5
        self.ele_y_arr = np.random.random(
            (self.hidden, )).astype("float32") - 0.5

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
    def _test_output(self):
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
            verify_node_count(program._graph, "fused_gemm_epilogue", 3),
            "[{}] The number of fused_gemm_epilogue is miss-matched in the computing graph.".
            format(type(self).__name__))
        act_fwd_name = self._get_act_type()[1]
        self.assertTrue(
            verify_node_count(program._graph, act_fwd_name, 1),
            "[{}] The number of {} is miss-matched in the computing graph.".
            format(type(self).__name__, act_fwd_name))

    def _pre_test_hooks(self):
        self.atol = 1e-4
        self.rtol = 1e-3

    def _get_act_type(self):
        return paddle.nn.ReLU, "relu"


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueReluFWDFP32(TestFuseGemmEpilogueFWDBase):
    def _pre_test_hooks(self):
        self.atol = 1e-3
        self.rtol = 1e-2

    def _get_act_type(self):
        return paddle.nn.ReLU, "relu"

    def test_output(self):
        self._test_output()


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueReluFWDFP16(TestFuseGemmEpilogueReluFWDFP32):
    def _pre_test_hooks(self):
        self.atol = 1e-3
        self.rtol = 1e-2

        fp16_var_list = paddle.static.amp.cast_model_to_fp16(self.main_prog)
        paddle.static.amp.cast_parameters_to_fp16(
            self.place, self.main_prog, to_fp16_var_names=fp16_var_list)

        self.data_arr = self.data_arr.astype("float16")
        self.matmul_y_arr = self.matmul_y_arr.astype("float16")
        self.ele_y_arr = self.ele_y_arr.astype("float16")


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGeluFWDFP32(TestFuseGemmEpilogueFWDBase):
    def _pre_test_hooks(self):
        self.atol = 1e-4
        self.rtol = 1e-3

    def _get_act_type(self):
        return paddle.nn.GELU, "gelu"

    def test_output(self):
        self._test_output()


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGeluFWDFP16(TestFuseGemmEpilogueGeluFWDFP32):
    def _pre_test_hooks(self):
        self.atol = 1e-3
        self.rtol = 1e-2

        fp16_var_list = paddle.static.amp.cast_model_to_fp16(self.main_prog)
        paddle.static.amp.cast_parameters_to_fp16(
            self.place, self.main_prog, to_fp16_var_names=fp16_var_list)

        self.data_arr = self.data_arr.astype("float16")
        self.matmul_y_arr = self.matmul_y_arr.astype("float16")
        self.ele_y_arr = self.ele_y_arr.astype("float16")


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueBWDBase(unittest.TestCase):
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

            multi_layer = MultiFCLayer(self.hidden, self._get_act_type()[0])
            with paddle.static.amp.fp16_guard():
                out = multi_layer(data, matmul_y, ele_y)
                self.loss = paddle.mean(out)
                paddle.static.append_backward(loss=self.loss)

        self.data_arr = np.random.random(
            (self.batch, self.seqlen, self.hidden)).astype("float32") - 0.5
        self.matmul_y_arr = np.random.random(
            (1, self.hidden, self.hidden)).astype("float32") - 0.5
        self.ele_y_arr = np.random.random(
            (self.hidden, )).astype("float32") - 0.5

        self.place = paddle.CUDAPlace(0)
        self.exe = paddle.static.Executor(self.place)
        self.exe.run(self.startup_prog)

        self._pre_test_hooks()

        self.feed = {
            "_data": self.data_arr,
            "_matmul_y": self.matmul_y_arr,
            "_ele_y": self.ele_y_arr
        }

        self.fetch = [
            self.loss.name,
            '{}.w_0@GRAD'.format(multi_layer.linear1.full_name()),
            '{}.b_0@GRAD'.format(multi_layer.linear1.full_name()),
            '{}.w_0@GRAD'.format(multi_layer.linear2.full_name()),
            '{}.b_0@GRAD'.format(multi_layer.linear2.full_name()),
            '{}.w_0@GRAD'.format(multi_layer.linear3.full_name()),
            '{}.b_0@GRAD'.format(multi_layer.linear3.full_name())
        ]
        self.outs_ref = self.exe.run(self.main_prog,
                                     feed=self.feed,
                                     fetch_list=self.fetch)

    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    def _test_output(self):
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.fuse_gemm_epilogue = True
        program = paddle.static.CompiledProgram(self.main_prog)
        program = program.with_data_parallel(
            loss_name=self.loss.name,
            build_strategy=build_strategy,
            places=paddle.static.cuda_places())

        outs_res = self.exe.run(program, feed=self.feed, fetch_list=self.fetch)

        for ref, res in zip(self.outs_ref, outs_res):
            self.assertTrue(
                compare(ref, res, self.atol, self.rtol),
                "[{}] output is miss-matched.".format(type(self).__name__))

        self.assertTrue(
            verify_node_count(program._graph, "fused_gemm_epilogue", 3),
            "[{}] The number of fused_gemm_epilogue is miss-matched in the computing graph.".
            format(type(self).__name__))
        self.assertTrue(
            verify_node_count(program._graph, "fused_gemm_epilogue_grad", 3),
            "[{}] The number of fused_gemm_epilogue_grad is miss-matched in the computing graph.".
            format(type(self).__name__))
        _, act_fwd_name, act_bwd_name = self._get_act_type()
        self.assertTrue(
            verify_node_count(program._graph, act_fwd_name, 1),
            "[{}] The number of {} is miss-matched in the computing graph.".
            format(type(self).__name__, act_fwd_name))
        self.assertTrue(
            verify_node_count(program._graph, act_bwd_name, 2),
            "[{}] The number of {} is miss-matched in the computing graph.".
            format(type(self).__name__, act_bwd_name))

    def _pre_test_hooks(self):
        self.atol = 1e-4
        self.rtol = 1e-3

    def _get_act_type(self):
        return paddle.nn.ReLU, "relu", "relu_grad"


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueReLUBWDFP32(TestFuseGemmEpilogueBWDBase):
    def _pre_test_hooks(self):
        self.atol = 1e-4
        self.rtol = 1e-3

    def _get_act_type(self):
        return paddle.nn.ReLU, "relu", "relu_grad"

    def test_output(self):
        self._test_output()


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueReLUBWDFP16(TestFuseGemmEpilogueReLUBWDFP32):
    def _pre_test_hooks(self):
        self.atol = 1e-3
        self.rtol = 1e-2

        fp16_var_list = paddle.static.amp.cast_model_to_fp16(self.main_prog)
        paddle.static.amp.cast_parameters_to_fp16(
            self.place, self.main_prog, to_fp16_var_names=fp16_var_list)

        self.data_arr = self.data_arr.astype("float16")
        self.matmul_y_arr = self.matmul_y_arr.astype("float16")
        self.ele_y_arr = self.ele_y_arr.astype("float16")


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGeLUBWDFP32(TestFuseGemmEpilogueBWDBase):
    def _pre_test_hooks(self):
        self.atol = 5e-4
        self.rtol = 1e-3

    def _get_act_type(self):
        return paddle.nn.GELU, "gelu", "gelu_grad"

    def test_output(self):
        self._test_output()


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGeLUBWDFP16(TestFuseGemmEpilogueGeLUBWDFP32):
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
    np.random.seed(0)
    unittest.main()
