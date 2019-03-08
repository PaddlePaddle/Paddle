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
import paddle.fluid.core as core
import paddle.fluid as fluid
import math
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
    def __init__(self, path_table, path_code, index):
        self.ptable_ = path_table
        self.pcode_ = path_code
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
            pre_output[i][j] += bias[idx][0]
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


def hsigmoidWithCustomTree(x, w, path_table, path_code, label, bias,
                           num_classes):
    batch_size = x.shape[0]
    code_length = len(path_table[0])
    code_table = [0 for _ in range(code_length)]
    # init pre_out with shape [N, code_length]
    pre_output = np.zeros((batch_size, code_length))
    pre_sum = np.zeros((batch_size, 1))
    out = np.zeros((batch_size, 1)).astype("float32")
    if isinstance(bias, np.ndarray):
        for i in range(batch_size):
            code_table = CodeTableWithCustomTree(path_table, path_code, i)
            length = code_table.get_length()
            for j in range(length):
                idx = code_table.cal_index(j)
                pre_output[i][j] += bias[idx][0]
    for i in range(batch_size):
        code_table = CodeTableWithCustomTree(path_table, path_code, i)
        length = code_table.get_length()
        for j in range(length):
            idx = code_table.cal_index(j)
            pre_output[i][j] += np.dot(w[idx], x[i])
    # clip[-40.0, 40.0]
    pre_output = np.clip(pre_output, -40.0, 40.0)
    # out(i, 0) = \sum_j  bit(i, j) * preout(i, j)
    for i in range(batch_size):
        code_table = CodeTableWithCustomTree(path_table, path_code, i)
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
        bias = np.random.random((num_classes - 1, 1)).astype("float32")
        self.attrs = {'num_classes': num_classes, 'is_sparse': False}
        self.inputs = {'X': x, 'W': w, 'Label': label, 'Bias': bias}
        pre_output, out = hsigmoid(x, w, label, bias, num_classes)
        self.outputs = {'PreOut': pre_output, 'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Bias', 'X', 'W'], ['Out'], no_grad_set=set('Label'))


class TestHSigmoidOpSparse(OpTest):
    def setUp(self):
        self.op_type = "hierarchical_sigmoid"
        num_classes = 6  #using 1,2,3,4,5,6 to build a huffman tree and select 1,2,5,6 as sample
        feature_size = 8
        batch_size = 4
        x = np.random.random((batch_size, feature_size)).astype("float32")
        w = np.random.random((num_classes - 1, feature_size)).astype("float32")
        label = np.array([0, 1, 4, 5])
        path_table = np.array(
            [(0, 2, -1, -1, -1), (0, 1, 3, -1, -1), (0, 1, 4, -1, -1),
             (0, 2, -1, -1,
              -1)])  #np.array to store 1,2,5,6s' non-leaf path(root -> leaf)
        path_code = np.array([(0, 0, -1, -1, -1), (1, 1, 1, -1, -1), (
            1, 0, 0, -1, -1), (0, 1, -1, -1, -1)])  #np.array to store 
        bias = np.random.random((num_classes - 1, 1)).astype("float32")
        self.attrs = {'num_classes': num_classes, 'is_sparse': True}
        self.inputs = {
            'X': x,
            'W': w,
            'PathTable': path_table,
            'PathCode': path_code,
            'Label': label,
            'Bias': bias
        }
        pre_output, out = hsigmoidWithCustomTree(x, w, path_table, path_code,
                                                 label, bias, num_classes)
        self.outputs = {'PreOut': pre_output, 'Out': out}

    def test_check_output(self):
        self.check_output()


class TestHSigmoidOpWithSparseGrad(unittest.TestCase):
    def hs_net_conf(self, is_sparse):
        input_word = fluid.layers.data(name="x", shape=[1], dtype='int64')
        path_table = fluid.layers.data(
            name='path_table', shape=[3], dtype='int64')
        path_code = fluid.layers.data(
            name='path_code', shape=[3], dtype='int64')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        data_list = [input_word, path_table, path_code, label]

        emb = fluid.layers.embedding(
            input=input_word,
            is_sparse=is_sparse,
            size=[3, 3],
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(3))))

        cost = fluid.layers.hsigmoid(
            input=emb,
            label=label,
            bias_attr=True,
            num_classes=3,
            path_table=path_table,
            path_code=path_code,
            is_custom=True,
            is_sparse=is_sparse)

        avg_cost = fluid.layers.reduce_mean(cost)

        return avg_cost, data_list

    def training_test(self, is_sparse):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            start_up = fluid.default_startup_program()
            start_up.random_seed = 1  # Fix random seed
            x = np.arange(6).reshape(6)
            path_table = np.array([(1, 2, -1), (1, 2, -1)])
            path_code = np.array([(1, 0, -1), (0, 0, -1)])
            label = np.array([1, 4])

            loss, data_list = self.hs_net_conf(is_sparse)
            optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
            optimizer.minimize(loss)

            main_program = fluid.default_main_program()
            place = fluid.CPUPlace()
            feeder = fluid.DataFeeder(feed_list=data_list, place=place)
            exe = fluid.Executor(place)

            exe.run(start_up)
            result = list()
            for i in range(10):
                data = [([[x[i % 2]]], [list(path_table[i % 2])],
                         [list(path_code[i % 2])], [label[i % 2]])]

                loss_val = exe.run(main_program,
                                   feed=feeder.feed(data),
                                   fetch_list=[loss])
                result.append(loss_val)
        return result

    def test_hs_grad_with_sparse(self):
        dense_result = self.training_test(is_sparse=False)
        sparse_result = self.training_test(is_sparse=True)
        assert (dense_result == sparse_result)


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
        path_table = np.array(
            [(0, 2, -1, -1, -1), (0, 1, 3, -1, -1), (0, 1, 4, -1, -1),
             (0, 2, -1, -1,
              -1)])  #np.array to store 1,2,5,6s' non-leaf path(root -> leaf)
        path_code = np.array([(0, 0, -1, -1, -1), (1, 1, 1, -1, -1), (
            1, 0, 0, -1, -1), (0, 1, -1, -1, -1)])  #np.array to store 
        bias = np.random.random((num_classes - 1, 1)).astype("float32")
        self.attrs = {'num_classes': num_classes, 'is_sparse': False}
        self.inputs = {
            'X': x,
            'W': w,
            'PathTable': path_table,
            'PathCode': path_code,
            'Label': label,
            'Bias': bias
        }
        pre_output, out = hsigmoidWithCustomTree(x, w, path_table, path_code,
                                                 label, bias, num_classes)
        self.outputs = {'PreOut': pre_output, 'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Bias', 'X', 'W'], ['Out'], no_grad_set=set('Label'))


class TestHSigmoidOpWithCostumTreeWithoutBias(OpTest):
    def setUp(self):
        self.op_type = "hierarchical_sigmoid"
        num_classes = 6  #using 1,2,3,4,5,6 to build a huffman tree and select 1,2,5,6 as sample
        feature_size = 8
        batch_size = 4
        x = np.random.random((batch_size, feature_size)).astype("float32") * 2
        w = np.random.random(
            (num_classes - 1, feature_size)).astype("float32") * 2
        label = np.array([0, 1, 4, 5])
        path_table = np.array(
            [(0, 2, -1, -1, -1), (0, 1, 3, -1, -1), (0, 1, 4, -1, -1),
             (0, 2, -1, -1,
              -1)])  #np.array to store 1,2,5,6s' non-leaf path(root -> leaf)
        path_code = np.array([(0, 0, -1, -1, -1), (1, 1, 1, -1, -1), (
            1, 0, 0, -1, -1), (0, 1, -1, -1, -1)])  #np.array to store 
        # bias = np.random.random((num_classes - 1, 1)).astype("float32")
        self.attrs = {'num_classes': num_classes, 'is_sparse': False}
        self.inputs = {
            'X': x,
            'W': w,
            'PathTable': path_table,
            'PathCode': path_code,
            'Label': label,
        }
        pre_output, out = hsigmoidWithCustomTree(
            x=x,
            w=w,
            path_table=path_table,
            path_code=path_code,
            label=label,
            bias=None,
            num_classes=num_classes)
        self.outputs = {'PreOut': pre_output, 'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'W'], ['Out'], no_grad_set=set('Label'))


if __name__ == '__main__':
    unittest.main()
