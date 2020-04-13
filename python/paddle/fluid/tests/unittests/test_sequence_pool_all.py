#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
from test_reorder_lod_tensor import convert_to_offset
import six
import random
from test_reorder_lod_tensor import convert_to_offset
import paddle.fluid.core as core


def compute_seqpool_sum(x, offset, out, pad_value=0.0):
    level = len(offset) - 1
    for i in range(len(offset[level]) - 1):
        if offset[level][i] == offset[level][i + 1]:
            out[i] = pad_value
        else:
            sub_x = x[offset[level][i]:offset[level][i + 1], :]
            out[i] = sub_x.sum(axis=0)


class TestSequencePoolAll(OpTest):
    def compute_np_out(self, all_vars):
        res = []
        for index in range(len(all_vars)):
            var = all_vars[index][0]
            lod = all_vars[index][1]
            level = len(lod) - 1
            offset = convert_to_offset(lod)
            out = np.zeros((len(lod[level]), self.feat_len)).astype(self.dtype)
            compute_seqpool_sum(var, offset, out, self.pad_value)
            res.append(out)
        return res

    def set_lod_data(self, var_num, batch_size, feat_len):
        res = []
        for index in six.moves.range(var_num):
            lod = []
            temp_lod = []
            for bid in range(batch_size):
                if bid % 2 == 0:
                    seq_len = random.randint(1, 5)
                else:
                    seq_len = random.randint(0, 4)
                temp_lod.append(seq_len)
            lod.append(temp_lod)
            x = np.random.uniform(
                0.1, 1, [np.sum(temp_lod), feat_len]).astype(self.dtype)
            res.append((x, lod))
        return res

    def config(self):
        self.var_num = 8
        self.batch_size = 100
        self.feat_len = 5
        self.var_names = [
            'x' + str(num) for num in six.moves.range(self.var_num)
        ]
        self.out_names = [
            'out' + str(num) for num in six.moves.range(self.var_num)
        ]
        self.dtype = "float64"
        self.pad_value = 0.0

    def setUp(self):
        self.op_type = "sequence_pool_all"
        self.config()
        self.vars = self.set_lod_data(self.var_num, self.batch_size,
                                      self.feat_len)
        np_out = self.compute_np_out(self.vars)
        self.inputs = {"X": list(zip(self.var_names, self.vars))}
        self.attrs = {"pad_value": 0.0, 'pooltype': "SUM"}
        self.outputs = {"Out": list(zip(self.out_names, np_out))}

    def test_check_output_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0), check_dygraph=False)

    def test_check_grad_gpu(self):
        if core.is_compiled_with_cuda():
            for index in range(self.var_num):
                self.check_grad_with_place(
                    core.CUDAPlace(0), [self.var_names[index]],
                    self.out_names[index],
                    check_dygraph=False)

    def test_check_output_cpu(self):
        try:
            self.check_output_with_place(
                place=core.CPUPlace(), check_dygraph=False)
        except:
            print("do not support cpu test, skip")


if __name__ == '__main__':
    unittest.main()
