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


def np_cal_batchfc(input, w, bias, batchcount):
    ins_num, _ = input.shape
    in_feat, w_col = w.shape
    out_feat = w_col / batchcount

    res = np.zeros((ins_num, w_col))
    for batch in range(batchcount):
        res[:, batch * out_feat:batch * out_feat + out_feat] = np.dot(
            input[:, in_feat * batch:in_feat * batch + in_feat],
            w[:, out_feat * batch:out_feat * batch + out_feat])

    for col in range(w_col):
        res[:, col] = res[:, col] + bias[0, col]
    return res


class TestBatchFCOp(OpTest):
    def config(self):
        self.batchcount = 10
        self.in_feat = 10
        self.out_feat = 10
        self.ins_num = 2
        self.dtype = "float64"

    def setUp(self):
        self.config()
        self.input = np.random.random(
            (self.ins_num, self.in_feat * self.batchcount)).astype(self.dtype)
        #self.input = np.ones((self.ins_num, self.in_feat * self.batchcount)).astype(self.dtype)
        #self.input = np.array(list(range(self.ins_num * self.in_feat * self.batchcount))).reshape([self.ins_num, self.in_feat * self.batchcount]).astype(self.dtype)
        #print(self.input)
        #self.input = np.ones((self.ins_num, self.in_feat * self.batchcount)).astype(self.dtype)
        self.w = np.random.random(
            (self.in_feat, self.out_feat * self.batchcount)).astype(self.dtype)
        #self.w = np.ones((self.in_feat, self.out_feat * self.batchcount)).astype(self.dtype)
        self.bias = np.random.random(
            (1, self.out_feat * self.batchcount)).astype(self.dtype)
        #self.bias = np.zeros((1,self.out_feat * self.batchcount)).astype(self.dtype)
        self.op_type = "batch_fc"
        np_out = np_cal_batchfc(self.input, self.w, self.bias, self.batchcount)
        np_out = np_out.astype(self.dtype)
        self.inputs = {"Input": self.input, "W": self.w, "Bias": self.bias}
        self.outputs = {"Out": np_out}
        self.attrs = {"batchcount": self.batchcount}

    def test_check_output_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(
                core.CUDAPlace(0), ["Bias", "W", "Input"], "Out")
            #core.CUDAPlace(0), ["Bias", "Input"], "Out")
            #core.CUDAPlace(0), ["Bias", "Input"], "Out")

        #class TestBatchFCOp1(OpTest):
        #    def config(self):
        #        self.slot_pairs_num = 10
        #        self.batch_size = 5
        #        self.in_dim = 10
        #        self.out_dim = 12
        #        self.dtype = "float64"
        #
        #    def setUp(self):
        #        self.config()
        #        self.input = np.random.random((self.slot_pairs_num, self.batch_size,
        #                                       self.in_dim)).astype(self.dtype)
        #        self.w = np.random.random((self.slot_pairs_num, self.in_dim,
        #                                   self.out_dim)).astype(self.dtype)
        #        self.bias = np.random.random((self.slot_pairs_num,
        #                                      self.out_dim)).astype(self.dtype)
        #        self.op_type = "batch_fc"
        #        np_out = np_cal_batchfc(self.input, self.w, self.bias)
        #        np_out = np_out.astype(self.dtype)
        #        self.inputs = {"Input": self.input, "W": self.w, "Bias": self.bias}
        #        self.outputs = {"Out": np_out}
        #
        #    def test_check_output_cpu(self):
        #        try:
        #            self.check_output_with_place(place=core.CPUPlace())
        #        except:
        #            print("do not support cpu test, skip")
        #
        #    def test_check_grad_cpu(self):
        #        try:
        #            self.check_grad_with_place(core.CPUPlace(), ["Bias", "W", "Input"],
        #                                       "Out")
        #        except:
        #            print("do not support cpu test, skip")
        #


if __name__ == "__main__":
    unittest.main()
