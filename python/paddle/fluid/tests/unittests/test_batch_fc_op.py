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
from paddle.fluid import Program, program_guard
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid.core as core


def np_cal_batchfc(input, w, bias, dorelu):
    slot_pairs_num, batch_size, in_dim = input.shape
    _, _, out_dim = w.shape
    res = np.zeros((slot_pairs_num, batch_size, out_dim))
    for slot in range(slot_pairs_num):
        res[slot, :] = np.dot(input[slot, :], w[slot, :])
    for slot in range(slot_pairs_num):
        for bindx in range(out_dim):
            res[slot, :, bindx] += bias[slot, bindx]
            if dorelu:
                res[slot, :, bindx]
    flag = (res > 0).astype(int)
    res = res * flag
    return res


class TestBatchFCOp(OpTest):
    def config(self):
        self.slot_pairs_num = 10
        self.batch_size = 100
        self.in_dim = 16
        self.out_dim = 32
        self.do_relu = True
        self.dtype = "float64"

    def setUp(self):
        self.config()
        self.input = np.random.random((self.slot_pairs_num, self.batch_size,
                                       self.in_dim)).astype(self.dtype)
        self.w = np.random.random((self.slot_pairs_num, self.in_dim,
                                   self.out_dim)).astype(self.dtype)

        self.bias = np.random.random((self.slot_pairs_num,
                                      self.out_dim)).astype(self.dtype)
        self.op_type = "batch_fc"
        np_out = np_cal_batchfc(self.input, self.w, self.bias, self.do_relu)
        np_out = np_out.astype(self.dtype)
        str_relu = ""
        if self.do_relu:
            str_relu = "relu"
        self.inputs = {"Input": self.input, "W": self.w, "Bias": self.bias}
        self.attrs = {'activation_type': str_relu}
        self.outputs = {"Out": np_out}

    def test_check_output_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0), check_dygraph=False)


#    def test_check_grad_gpu(self):
#        if core.is_compiled_with_cuda():
#            self.check_grad_with_place(core.CUDAPlace(0), ["RankParam"], "Out")


class TestBatchFCOp1(TestBatchFCOp):
    def config(self):
        self.slot_pairs_num = 10
        self.batch_size = 100
        self.in_dim = 16
        self.out_dim = 32
        self.do_relu = False
        self.dtype = "float64"


if __name__ == "__main__":
    unittest.main()
