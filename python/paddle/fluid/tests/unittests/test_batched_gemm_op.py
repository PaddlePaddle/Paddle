# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle.fluid as fluid
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid.core as core


def np_batched_gemm(x, y, batch_count, mat_m, mat_n, mat_k):
    res = np.zeros((batch_count, mat_m, mat_n))
    for ins in range(batch_count):
        res[ins, :] = np.dot(x[ins, :], y[ins, :])
    return res


class TestBatchedGemmOp1(OpTest):
    def config(self):
        self.batch_count = 100
        self.mat_m = 8
        self.mat_n = 12
        self.mat_k = 5
        self.dtype = "float64"

    def setUp(self):
        self.op_type = "batched_gemm"
        self.config()
        x = np.random.random(
            (self.batch_count, self.mat_m, self.mat_k)).astype(self.dtype)
        y = np.random.random(
            (self.batch_count, self.mat_k, self.mat_n)).astype(self.dtype)
        np_res = np_batched_gemm(x, y, self.batch_count, self.mat_m, self.mat_n,
                                 self.mat_k)

        self.inputs = {
            "X": x,
            "Y": y,
        }
        self.attrs = {
            "BatchCount": self.batch_count,
            "Mat_M": self.mat_m,
            "Mat_N": self.mat_n,
            "Mat_K": self.mat_k
        }
        self.outputs = {"Out": np_res}

    def test_check_output_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(core.CUDAPlace(0), ["X", "Y"], "Out")


class TestBatchedGemmOp2(OpTest):
    def config(self):
        self.batch_count = 100
        self.mat_m = 8
        self.mat_n = 12
        self.mat_k = 5
        self.dtype = "float64"

    def setUp(self):
        self.op_type = "batched_gemm"
        self.config()
        x = np.random.random(
            (self.batch_count, self.mat_m, self.mat_k)).astype(self.dtype)
        y = np.random.random(
            (self.batch_count, self.mat_k, self.mat_n)).astype(self.dtype)
        np_res = np_batched_gemm(x, y, self.batch_count, self.mat_m, self.mat_n,
                                 self.mat_k)

        self.inputs = {
            "X": x,
            "Y": y,
        }
        self.attrs = {
            "BatchCount": self.batch_count,
            "Mat_M": self.mat_m,
            "Mat_N": self.mat_n,
            "Mat_K": self.mat_k
        }
        self.outputs = {"Out": np_res}

    def test_check_output_cpu(self):
        try:
            self.check_output_with_place(place=core.CPUPlace())
        except:
            print("do not support cpu test, skip")

    def test_check_grad_cpu(self):
        try:
            self.check_grad_with_place(core.CPUPlace(), ["RankParam"], "Out")
        except:
            print("do not support cpu test, skip")


if __name__ == "__main__":
    unittest.main()
