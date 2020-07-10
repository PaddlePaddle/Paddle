#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest


class TestCrossNormHadamardOp(OpTest):
    """
    test forward and backward
    """

    def setUp(self):
        self.op_type = 'cross_norm_hadamard'

        ins_num = 100
        embed_dim = 2
        fields_num = 5
        tp = np.float64

        input_a = np.random.random([ins_num, embed_dim]).astype(tp)
        input_b = np.random.random([ins_num, embed_dim]).astype(tp)
        input = np.concatenate((input_a, input_b), axis=1)
        input_multi = input_a * input_b
        input_sim = np.sum(input_multi, axis=1, keepdims=True)

        np_res = np.concatenate(
            (input_a, input_b, input_multi, input_sim), axis=1)

        for _ in range(fields_num - 1):
            input_a = np.random.random([ins_num, embed_dim]).astype(tp)
            input_b = np.random.random([ins_num, embed_dim]).astype(tp)
            input = np.concatenate((input, input_a, input_b), axis=1)
            input_multi = input_a * input_b
            input_sim = np.sum(input_multi, axis=1, keepdims=True)

            np_res = np.concatenate(
                (np_res, input_a, input_b, input_multi, input_sim), axis=1)

        summary_input = np.zeros(
            [3, (embed_dim * 3 + 1) * fields_num]).astype(tp)
        summary_input[0, :] = 1e4
        summary_input[1, :] = 0.0
        summary_input[2, :] = 1e4

        np_mean = summary_input[1, :] / summary_input[0, :]
        np_scale = np.sqrt(summary_input[0, :] / summary_input[2, :])

        self.inputs = {"Input": input, "SummaryInput": summary_input}
        self.outputs = {
            "Out": np_res,
            "CudaMeans": np_mean,
            "CudaScales": np_scale
        }
        self.attrs = {"fields_num": fields_num, "embed_dim": embed_dim}

    def test_check_output_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(core.CUDAPlace(0), ["Input"], "Out")


class TestRankAttentionOpCpu(OpTest):
    def setUp(self):
        """
        """
        self.op_type = 'cross_norm_hadamard'

        ins_num = 100
        embed_dim = 2
        fields_num = 5
        tp = np.float64

        input_a = np.random.random([ins_num, embed_dim]).astype(tp)
        input_b = np.random.random([ins_num, embed_dim]).astype(tp)
        input = np.concatenate((input_a, input_b), axis=1)
        input_multi = input_a * input_b
        input_sim = np.sum(input_multi, axis=1, keepdims=True)

        np_res = np.concatenate(
            (input_a, input_b, input_multi, input_sim), axis=1)

        for _ in range(fields_num - 1):
            input_a = np.random.random([ins_num, embed_dim]).astype(tp)
            input_b = np.random.random([ins_num, embed_dim]).astype(tp)
            input = np.concatenate((input, input_a, input_b), axis=1)
            input_multi = input_a * input_b
            input_sim = np.sum(input_multi, axis=1, keepdims=True)

            np_res = np.concatenate(
                (np_res, input_a, input_b, input_multi, input_sim), axis=1)

        summary_input = np.zeros(
            [3, (embed_dim * 3 + 1) * fields_num]).astype(tp)
        summary_input[0, :] = 1e4
        summary_input[1, :] = 0.0
        summary_input[2, :] = 1e4

        np_mean = summary_input[1, :] / summary_input[0, :]
        np_scale = np.sqrt(summary_input[0, :] / summary_input[2, :])

        self.inputs = {"Input": input, "SummaryInput": summary_input}
        self.outputs = {
            "Out": np_res,
            "CudaMeans": np_mean,
            "CudaScales": np_scale
        }
        self.attrs = {"fields_num": fields_num, "embed_dim": embed_dim}

    def test_check_output_cpu(self):
        try:
            self.check_output_with_place(place=core.CPUPlace())
        except:
            print("do not support cpu test, skip")

    def test_check_grad_cpu(self):
        try:
            self.check_grad_with_place(core.CPUPlace(), ["Input"], "Out")
        except:
            print("do not support cpu test, skip")


if __name__ == '__main__':
    unittest.main()
