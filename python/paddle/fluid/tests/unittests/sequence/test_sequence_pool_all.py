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
import six
import random
import numpy as np
import sys
sys.path.append("../")
from op_test import OpTest
import paddle.fluid.core as core
from test_reorder_lod_tensor import convert_to_offset


def compute_seqpool_sum(x, offset, out, pad_value=0.0):
    level = len(offset) - 1
    for i in range(len(offset[level]) - 1):
        if offset[level][i] == offset[level][i + 1]:
            out[i] = pad_value
        else:
            sub_x = x[offset[level][i]:offset[level][i + 1], :]
            out[i] = sub_x.sum(axis=0)


class TestSequencePoolAll1(OpTest):
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
        self.feat_len = 2
        self.var_names = [
            'x' + str(num) for num in six.moves.range(self.var_num)
        ]
        self.out_names = [
            'out' + str(num) for num in six.moves.range(self.var_num)
        ]
        self.dtype = "float64"
        self.pad_value = 0.0

    def setUp(self):
        self.__class__.op_type = "sequence_pool_all"
        self.op_type = self.__class__.op_type
        self.config()
        self.vars = self.set_lod_data(self.var_num, self.batch_size,
                                      self.feat_len)
        np_out = self.compute_np_out(self.vars)
        self.inputs = {"X": list(zip(self.var_names, self.vars))}
        self.attrs = {"pad_value": 0.0, 'pooltype': "SUM"}
        self.outputs = {"Out": list(zip(self.out_names, np_out))}

    def test_check_output_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad_gpu(self):
        if core.is_compiled_with_cuda():
            for index in range(self.var_num):
                self.check_grad_with_place(
                    core.CUDAPlace(0), [self.var_names[index]],
                    self.out_names[index])


class TestSequencePoolAll2(TestSequencePoolAll1):
    def config(self):
        self.var_num = 5
        self.batch_size = 1
        self.feat_len = 100
        self.var_names = [
            'x' + str(num) for num in six.moves.range(self.var_num)
        ]
        self.out_names = [
            'out' + str(num) for num in six.moves.range(self.var_num)
        ]
        self.dtype = "float32"
        self.pad_value = 0.5

    def set_lod_data(self, var_num, batch_size, feat_len):
        res = []
        for index in six.moves.range(var_num):
            lod = []
            seq_len = random.randint(1, 5)
            temp_lod = [seq_len]
            lod.append(temp_lod)
            lodtensor = self.get_sequence_batch_size_1_input(
                lod=lod, shape=[lod[0][0], self.feat_len])
            res.append(lodtensor)
        return res


class TestSequencePoolAll3(TestSequencePoolAll1):
    def config(self):
        self.var_num = 5
        self.feat_len = 100
        self.batch_size = 9
        self.var_names = [
            'x' + str(num) for num in six.moves.range(self.var_num)
        ]
        self.out_names = [
            'out' + str(num) for num in six.moves.range(self.var_num)
        ]
        self.dtype = "float32"
        self.pad_value = 0.0

    def set_lod_data(self, var_num, batch_size, feat_len):
        res = []
        lod = [[0, 0, 4, 0, 3, 0, 0, 5, 0, 0]]
        shape = [12, self.feat_len]
        for index in six.moves.range(var_num):
            lodtensor = self.get_sequence_instance_size_0_input(lod, shape)
            res.append(lodtensor)
        return res


class TestSequencePoolAll4(TestSequencePoolAll1):
    """ checkout only cpu"""

    def config(self):
        self.var_num = 5
        self.feat_len = 100
        self.batch_size = 9
        self.var_names = [
            'x' + str(num) for num in six.moves.range(self.var_num)
        ]
        self.out_names = [
            'out' + str(num) for num in six.moves.range(self.var_num)
        ]
        self.dtype = "float32"
        self.pad_value = 0.0

    def set_lod_data(self, var_num, batch_size, feat_len):
        res = []
        lod = [[0, 0, 4, 0, 3, 0, 0, 5, 0, 0]]
        shape = [12, self.feat_len]
        for index in six.moves.range(var_num):
            lodtensor = self.get_sequence_instance_size_0_input(lod, shape)
            res.append(lodtensor)
        return res

    def test_check_output_cpu(self):
        try:
            self.check_output_with_place(place=core.CPUPlace())
        except:
            print("do not support cpu test, skip")

    def test_check_grad_cpu(self):
        try:
            for index in range(self.var_num):
                self.check_grad_with_place(core.CPUPlace(),
                                           [self.var_names[index]],
                                           self.out_names[index])
        except:
            print("do not support cpu test, skip")

    def test_check_output_gpu(self):
        pass

    def test_check_grad_gpu(self):
        pass


if __name__ == '__main__':
    unittest.main()
