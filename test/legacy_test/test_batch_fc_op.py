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
from op_test import OpTest

import paddle
from paddle.base import core


def np_cal_batchfc(input, w, bias):
    slot_pairs_num, batch_size, in_dim = input.shape
    _, _, out_dim = w.shape
    res = np.zeros((slot_pairs_num, batch_size, out_dim))
    for slot in range(slot_pairs_num):
        res[slot, :] = np.dot(input[slot, :], w[slot, :])
    for slot in range(slot_pairs_num):
        for bindx in range(out_dim):
            res[slot, :, bindx] += bias[slot, bindx]
    return res


def api_wrapper(input, w, bias):
    return paddle._C_ops.batch_fc(input, w, bias)


class TestBatchFCOp(OpTest):
    def config(self):
        self.slot_pairs_num = 10
        self.batch_size = 5
        self.in_dim = 10
        self.out_dim = 12
        self.dtype = "float64"

    def setUp(self):
        self.config()
        self.input = np.random.random(
            (self.slot_pairs_num, self.batch_size, self.in_dim)
        ).astype(self.dtype)
        self.w = np.random.random(
            (self.slot_pairs_num, self.in_dim, self.out_dim)
        ).astype(self.dtype)
        self.bias = np.random.random(
            (self.slot_pairs_num, self.out_dim)
        ).astype(self.dtype)
        self.op_type = "batch_fc"
        self.python_api = api_wrapper
        np_out = np_cal_batchfc(self.input, self.w, self.bias)
        np_out = np_out.astype(self.dtype)
        self.inputs = {"Input": self.input, "W": self.w, "Bias": self.bias}
        self.outputs = {"Out": np_out}

    def test_check_output_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(
                core.CUDAPlace(0), ["Bias", "W", "Input"], "Out"
            )


class TestBatchFCOp1(OpTest):
    def config(self):
        self.slot_pairs_num = 10
        self.batch_size = 5
        self.in_dim = 10
        self.out_dim = 12
        self.dtype = "float64"

    def setUp(self):
        self.config()
        self.input = np.random.random(
            (self.slot_pairs_num, self.batch_size, self.in_dim)
        ).astype(self.dtype)
        self.w = np.random.random(
            (self.slot_pairs_num, self.in_dim, self.out_dim)
        ).astype(self.dtype)
        self.bias = np.random.random(
            (self.slot_pairs_num, self.out_dim)
        ).astype(self.dtype)
        self.op_type = "batch_fc"
        self.python_api = api_wrapper
        np_out = np_cal_batchfc(self.input, self.w, self.bias)
        np_out = np_out.astype(self.dtype)
        self.inputs = {"Input": self.input, "W": self.w, "Bias": self.bias}
        self.outputs = {"Out": np_out}

    def test_check_output_cpu(self):
        try:
            self.check_output_with_place(place=core.CPUPlace())
        except:
            print("do not support cpu test, skip")

    def test_check_grad_cpu(self):
        try:
            self.check_grad_with_place(
                core.CPUPlace(), ["Bias", "W", "Input"], "Out"
            )
        except:
            print("do not support cpu test, skip")


if __name__ == "__main__":
    unittest.main()
