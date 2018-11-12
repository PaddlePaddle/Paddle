#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import math
# import paddle.fluid as fluid
# import paddle.fluid.core as core
# from op_builder import OpBuilder
from op_test import OpTest

np.random.seed(100)


def find_latest_set(num):
    return 1 + int(math.floor(math.log(num, 2)))


class CodeTable(object):
    def __init__(self, num_classes, code):
        self.c = num_classes + code

    def cal_index(self, bit):
        return (self.c >> (bit + 1)) - 1

    def get_length(self):
        return find_latest_set(self.c) - 1

    def cal_bit(self, bit):
        return self.c & (1 << bit)


class CodeTableWithCustomTree(object):
    def __init__(self, ptable, pcode, index):
        self.ptable_ = ptable
        self.pcode_ = pcode
        self.index_ = index

    def cal_index(self, bit):
        return self.ptable_[self.index_][bit]

    def get_length(self):
        length = 0
        for ele in self.ptable_[self.index_]:  # find the first -1 to stop trace

            if ele >= 0:
                length = length + 1
            else:
                return length
        return length

    def cal_bit(self, bit):
        return self.pcode_[self.index_][bit]


def hsigmoid(x, w, label, bias, num_classes):
    batch_size = x.shape[0]
    code_length = find_latest_set(num_classes - 1)
    code_table = [0 for _ in range(code_length)]
    pre_output = np.zeros((batch_size, code_length))
    pre_sum = np.zeros((batch_size, 1))
    out = np.zeros((batch_size, 1)).astype("float32")
    for i in range(batch_size):
        code_table = CodeTable(num_classes, label[i])
        length = code_table.get_length()
        for j in range(length):
            idx = code_table.cal_index(j)
            pre_output[i][j] += bias[0][idx]
    for i in range(batch_size):
        code_table = CodeTable(num_classes, label[i])
        length = code_table.get_length()
        for j in range(length):
            idx = code_table.cal_index(j)
            pre_output[i][j] += np.dot(w[idx], x[i])
    # clip[-40.0, 40.0]
    pre_output = np.clip(pre_output, -40.0, 40.0)
    # out(i, 0) = \sum_j  bit(i, j) * preout(i, j)
    for i in range(batch_size):
        code_table = CodeTable(num_classes, label[i])
        length = code_table.get_length()
        sum = 0.0
        for j in range(length):
            if code_table.cal_bit(j):
                sum += pre_output[i][j]
        out[i] = -1.0 * sum
    # soft relu
    pre_output = np.log(1 + np.exp(pre_output))
    pre_sum = pre_output.sum(1).reshape((batch_size, 1))
    out += pre_sum
    return pre_output, out


def hsigmoidWithCustomTree(x, w, ptable, pcode, label, bias, num_classes):
    batch_size = x.shape[0]
    code_length = len(ptable[0])
    code_table = [0 for _ in range(code_length)]
    # init pre_out with shape [N, code_length]
    pre_output = np.zeros((batch_size, code_length))
    pre_sum = np.zeros((batch_size, 1))
    out = np.zeros((batch_size, 1)).astype("float32")
    for i in range(batch_size):
        code_table = CodeTableWithCustomTree(ptable, pcode, i)
        length = code_table.get_length()
        for j in range(length):
            idx = code_table.cal_index(j)
            pre_output[i][j] += bias[0][idx]
    for i in range(batch_size):
        code_table = CodeTableWithCustomTree(ptable, pcode, i)
        length = code_table.get_length()
        for j in range(length):
            idx = code_table.cal_index(j)
            pre_output[i][j] += np.dot(w[idx], x[i])
    # clip[-40.0, 40.0]
    pre_output = np.clip(pre_output, -40.0, 40.0)
    # out(i, 0) = \sum_j  bit(i, j) * preout(i, j)
    for i in range(batch_size):
        code_table = CodeTableWithCustomTree(ptable, pcode, i)
        length = code_table.get_length()
        sum = 0.0
        for j in range(length):
            if code_table.cal_bit(j):
                sum += pre_output[i][j]
        out[i] = -1.0 * sum
    # soft relu
    pre_output = np.log(1 + np.exp(pre_output))
    pre_sum = pre_output.sum(1).reshape((batch_size, 1))
    out += pre_sum
    return pre_output, out


class TestHSigmoidOp(OpTest):
    def setUp(self):
        self.op_type = "hierarchical_sigmoid"
        num_classes = 6
        feature_size = 8
        batch_size = 4
        x = np.random.random((batch_size, feature_size)).astype("float32") * 2
        w = np.random.random(
            (num_classes - 1, feature_size)).astype("float32") * 2
        label = np.random.randint(0, num_classes, (batch_size, 1))
        bias = np.random.random((1, num_classes - 1)).astype("float32")
        self.attrs = {'num_classes': num_classes}
        self.inputs = {'X': x, 'W': w, 'Label': label, 'Bias': bias}
        pre_output, out = hsigmoid(x, w, label, bias, num_classes)
        self.outputs = {'PreOut': pre_output, 'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Bias', 'X', 'W'], ['Out'], no_grad_set=set('Label'))


class TestHSigmoidOpWithCostumTree(OpTest):
    def setUp(self):
        self.op_type = "hierarchical_sigmoid"
        num_classes = 6  #using 1,2,3,4,5,6 to build a huffman tree and select 1,2,5,6 as sample
        feature_size = 8
        batch_size = 4
        x = np.random.random((batch_size, feature_size)).astype("float32") * 2
        w = np.random.random(
            (num_classes - 1, feature_size)).astype("float32") * 2
        label = np.array([0, 1, 4, 5])
        ptable = np.array(
            [(0, 2, -1, -1, -1), (0, 1, 3, -1, -1), (0, 1, 4, -1, -1),
             (0, 2, -1, -1,
              -1)])  #np.array to store 1,2,5,6s' non-leaf path(root -> leaf)
        pcode = np.array([(0, 0, -1, -1, -1), (1, 1, 1, -1, -1), (
            1, 0, 0, -1, -1), (0, 1, -1, -1, -1)])  #np.array to store 
        bias = np.random.random((1, num_classes - 1)).astype("float32")
        self.attrs = {'num_classes': num_classes}
        self.inputs = {
            'X': x,
            'W': w,
            'PTable': ptable,
            'PCode': pcode,
            'Label': label,
            'Bias': bias
        }
        pre_output, out = hsigmoidWithCustomTree(x, w, ptable, pcode, label,
                                                 bias, num_classes)
        self.outputs = {'PreOut': pre_output, 'Out': out}

    def test_check_output(self):
        print("checking output in CostumTree")
        self.check_output()

    def test_check_grad(self):
        print("checking outputGrad in CostumTree")
        self.check_grad(['Bias', 'X', 'W'], ['Out'], no_grad_set=set('Label'))


if __name__ == '__main__':
    unittest.main()
