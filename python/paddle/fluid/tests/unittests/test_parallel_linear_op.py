# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import random
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid.core as core


def np_cal_parallel_linear(input, w, bias, expert_count):
    batch_size, in_dim = input.shape
    expert_num, _, out_dim = w.shape

    res = np.zeros((batch_size, out_dim))

    ptr = 0
    for i in range(len(expert_count)):
        if expert_count[i] == 0:
            continue
        x = input[ptr:ptr + expert_count[i], :]
        y = w[i, :]
        res[ptr:ptr + expert_count[i], :] = np.dot(x, y)
        res[ptr:ptr + expert_count[i], :] += bias[i, :]
        ptr = ptr + expert_count[i]

    return res


class TestParallelLinearOp(OpTest):
    def config(self):
        self.in_dim = 10
        self.out_dim = 20
        self.dtype = "float64"

    def setUp(self):
        self.config()
        self.expert_count = np.array(
            [2, 0, 1, 2, 3, 0, 0, 0, 2]).astype(np.int64)
        self.expert_num = len(self.expert_count)

        self.batch_size = np.sum(self.expert_count)
        self.input = np.random.random(
            (self.batch_size, self.in_dim)).astype(self.dtype)
        self.w = np.random.random(
            (self.expert_num, self.in_dim, self.out_dim)).astype(self.dtype)
        self.bias = np.random.random(
            (self.expert_num, self.out_dim)).astype(self.dtype)

        self.op_type = "parallel_linear"

        np_out = np_cal_parallel_linear(self.input, self.w, self.bias,
                                        self.expert_count)
        np_out = np_out.astype(self.dtype)

        # self.inputs = {"X": self.input, "W": self.w, "Bias": self.bias, 'Expert_Count': self.expert_count}
        self.inputs = {"X": self.input, "W": self.w, "Bias": self.bias}

        self.attrs = {
            'expert_count': self.expert_count,
            # 'num_columns': 319,
            # 'dtype': framework.convert_np_dtype_to_dtype_(np.int32)
        }
        self.outputs = {"Out": np_out}

    def test_check_output_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(
                core.CUDAPlace(0), ["Bias", "W", "X"], "Out")


if __name__ == "__main__":
    unittest.main()
