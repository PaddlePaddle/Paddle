# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""This is unit test of Test filter_instag Op."""

import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from op_test import OpTest
import random
from decorator_helper import prog_scope
from paddle.fluid.op import Operator
"""This is Test Case 1"""


class TestFilterByInstagOp(OpTest):

    def setUp(self):
        self.op_type = 'filter_by_instag'
        x1 = np.zeros((36, 4), dtype=np.float64)
        for i in range(36):
            for j in range(4):
                x1[i, j] = i
        x1_lod = [[1, 2, 3, 4, 5, 6, 7, 8]]

        x2 = np.array([[1], [2], [1], [2], [1], [2], [1], [2]]).astype('int64')
        x2_lod = [[1, 1, 1, 1, 1, 1, 1, 1]]

        x3 = np.array([2]).astype('int64')

        out = np.zeros((20, 4), dtype=np.float64)
        out_lod = [[2, 4, 6, 8]]
        start_num_lst = [1, 6, 15, 28]

        ln = 0
        for i in range(4):
            start = start_num_lst[i]
            len = out_lod[0][i]
            for j in range(len):
                cur = start + j
                for k in range(4):
                    out[ln, k] = cur
                ln += 1

        mmap = np.array([[0, 1, 2], [2, 6, 4], [6, 15, 6], [12, 28,
                                                            8]]).astype('int64')
        mmap_lod = [[1, 1, 1, 1]]

        loss_weight = np.array([[1], [1], [1], [1]]).astype('double')

        self.inputs = {
            'Ins': (x1, x1_lod),
            'Ins_tag': (x2, x2_lod),
            'Filter_tag': x3,
        }
        self.outputs = {
            'Out': (out, out_lod),
            'LossWeight': (loss_weight, mmap_lod),
            'IndexMap': (mmap, mmap_lod)
        }

        self.attrs = {'is_lod': True, 'out_val_if_empty': 0}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Ins'],
                        'Out',
                        no_grad_set=set(['Ins_tag', 'Filter_tag']))


"""This is Test Case 2"""


class TestFilterByInstagOp2(OpTest):

    def setUp(self):
        self.op_type = 'filter_by_instag'

        x1 = np.random.random((4, 36)).astype('double')
        x1_lod = [[1, 1, 1, 1]]

        x2 = np.array([[2], [1], [2], [1]]).astype('int64')
        x2_lod = [[1, 1, 1, 1]]

        x3 = np.array([1]).astype('int64')

        out = np.zeros([2, 36]).astype('double')
        out[0] = x1[1]
        out[1] = x1[3]
        out_lod = [[1, 1]]

        mmap = np.array([[0, 1, 1], [1, 3, 1]]).astype('int64')
        mmap_lod = [[1, 1]]

        loss_weight = np.array([[1], [1]]).astype('double')
        self.inputs = {
            'Ins': (x1, x1_lod),
            'Ins_tag': (x2, x2_lod),
            'Filter_tag': x3,
        }

        self.outputs = {
            'Out': (out, out_lod),
            'LossWeight': (loss_weight, mmap_lod),
            'IndexMap': (mmap, mmap_lod)
        }
        self.attrs = {'is_lod': True, 'out_val_if_empty': 0}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Ins'],
                        'Out',
                        no_grad_set=set(['Ins_tag', 'Filter_tag']))


"""This is Test Case 3"""


class TestFilterByInstagOp3(OpTest):

    def setUp(self):
        self.op_type = 'filter_by_instag'

        x1 = np.random.random((4, 36)).astype('double')
        x1_lod = [[1, 1, 1, 1]]

        x2 = np.array([[2], [1], [2], [1]]).astype('int64')
        x2_lod = [[1, 1, 1, 1]]

        x3 = np.array([3]).astype('int64')

        out = np.zeros((1, 36)).astype('double')
        out_lod = [[1]]

        mmap = np.array([[0, 1, 1]]).astype('int64')
        mmap_lod = [[1]]

        loss_weight = np.array([[0]]).astype('double')
        self.inputs = {
            'Ins': (x1, x1_lod),
            'Ins_tag': (x2, x2_lod),
            'Filter_tag': x3,
        }
        self.outputs = {
            'Out': (out, out_lod),
            'LossWeight': (loss_weight, mmap_lod),
            'IndexMap': (mmap, mmap_lod)
        }
        self.attrs = {'is_lod': True, 'out_val_if_empty': 0}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Ins'],
                        'Out',
                        no_grad_set=set(['Ins_tag', 'Filter_tag']))


"""This is Test Case 4"""


class TestFilterByInstagOp4(OpTest):

    def setUp(self):
        self.op_type = 'filter_by_instag'

        x1 = np.random.random((4, 36)).astype('double')

        x2 = np.array([[2], [1], [2], [1]]).astype('int64')
        x2_lod = [[1, 1, 1, 1]]

        x3 = np.array([3]).astype('int64')

        out = np.zeros((1, 36)).astype('double')
        out_lod = [[1]]

        mmap = np.array([[0, 1, 1]]).astype('int64')
        mmap_lod = [[1]]

        loss_weight = np.array([[0]]).astype('double')
        self.inputs = {
            'Ins': x1,
            'Ins_tag': (x2, x2_lod),
            'Filter_tag': x3,
        }
        self.outputs = {
            'Out': (out, out_lod),
            'LossWeight': (loss_weight, mmap_lod),
            'IndexMap': (mmap, mmap_lod)
        }
        self.attrs = {'is_lod': False, 'out_val_if_empty': 0}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Ins'],
                        'Out',
                        no_grad_set=set(['Ins_tag', 'Filter_tag']))


class TestFilterByInstagOp6(OpTest):

    def setUp(self):
        self.op_type = 'filter_by_instag'

        x1 = np.random.random((4, 36)).astype('int64')

        x2 = np.array([[2], [1], [2], [1]]).astype('int64')
        x2_lod = [[1, 1, 1, 1]]

        x3 = np.array([3]).astype('int64')

        out = np.zeros((1, 36)).astype('double')
        out_lod = [[1]]

        mmap = np.array([[0, 1, 1]]).astype('int64')
        mmap_lod = [[1]]

        loss_weight = np.array([[0]]).astype('double')
        self.inputs = {
            'Ins': x1,
            'Ins_tag': (x2, x2_lod),
            'Filter_tag': x3,
        }
        self.outputs = {
            'Out': (out, out_lod),
            'LossWeight': (loss_weight, mmap_lod),
            'IndexMap': (mmap, mmap_lod)
        }
        self.attrs = {'is_lod': False, 'out_val_if_empty': 0}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass


class TestFilterByInstagOp7(OpTest):

    def setUp(self):
        self.op_type = 'filter_by_instag'

        x1 = np.random.random((4, 36)).astype('int32')

        x2 = np.array([[2], [1], [2], [1]]).astype('int64')
        x2_lod = [[1, 1, 1, 1]]

        x3 = np.array([3]).astype('int64')

        out = np.zeros((1, 36)).astype('double')
        out_lod = [[1]]

        mmap = np.array([[0, 1, 1]]).astype('int64')
        mmap_lod = [[1]]

        loss_weight = np.array([[0]]).astype('double')
        self.inputs = {
            'Ins': x1,
            'Ins_tag': (x2, x2_lod),
            'Filter_tag': x3,
        }
        self.outputs = {
            'Out': (out, out_lod),
            'LossWeight': (loss_weight, mmap_lod),
            'IndexMap': (mmap, mmap_lod)
        }
        self.attrs = {'is_lod': False, 'out_val_if_empty': 0}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass


if __name__ == '__main__':
    unittest.main()
