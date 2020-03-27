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
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid.layers as layers
import paddle.fluid as fluid
import random
import six


class TestTDMChildOp(OpTest):
    def setUp(self):
        self.__class__.op_type = "tdm_child"
        self.config()
        tree_info = self.create_tdm_tree()
        tree_info_np = np.array(tree_info).astype(self.info_type)

        x_np = np.random.randint(
            low=0, high=26, size=self.x_shape).astype(self.x_type)
        children_res = []
        leaf_mask_res = []
        for batch in x_np:
            for node in batch:
                children = []
                if node != 0:
                    children.append(tree_info[node][3])
                    children.append(tree_info[node][4])
                else:
                    children.append(0)
                    children.append(0)
                mask = []
                for child in children:
                    m = int(tree_info[child][0] != 0)
                    mask.append(m)
                children_res += children
                leaf_mask_res += mask
        children_res_np = np.array(children_res).astype(self.info_type)
        leaf_mask_res_np = np.array(leaf_mask_res).astype(self.info_type)

        child = np.reshape(children_res_np, self.child_shape)
        leaf_mask = np.reshape(leaf_mask_res_np, self.child_shape)

        self.attrs = {'Child_nums': 2}
        self.inputs = {'X': x_np, 'Tree_info': tree_info_np}
        self.outputs = {'Child': child, 'Leaf_mask': leaf_mask}

    def create_tdm_tree(self):
        """Create tdm tree info"""
        tree_info = [
            [0, 0, 0, 1, 2],
            [0, 1, 0, 3, 4],
            [0, 1, 0, 5, 6],
            [0, 2, 1, 7, 8],
            [0, 2, 1, 9, 10],
            [0, 2, 2, 11, 12],
            [0, 2, 2, 13, 0],
            [0, 3, 3, 14, 15],
            [0, 3, 3, 16, 17],
            [0, 3, 4, 18, 19],
            [0, 3, 4, 20, 21],
            [0, 3, 5, 22, 23],
            [0, 3, 5, 24, 25],
            [12, 3, 6, 0, 0],
            [0, 4, 7, 0, 0],
            [1, 4, 7, 0, 0],
            [2, 4, 8, 0, 0],
            [3, 4, 8, 0, 0],
            [4, 4, 9, 0, 0],
            [5, 4, 9, 0, 0],
            [6, 4, 10, 0, 0],
            [7, 4, 10, 0, 0],
            [8, 4, 11, 0, 0],
            [9, 4, 11, 0, 0],
            [10, 4, 12, 0, 0],
            [11, 4, 12, 0, 0],
        ]
        return tree_info

    def config(self):
        """set test shape & type"""
        self.x_shape = (10, 20)
        self.child_shape = (10, 20, 2)
        self.x_type = 'int32'
        self.info_type = 'int32'

    def test_check_output(self):
        self.check_output()


class TestCase1(TestTDMChildOp):
    def config(self):
        """check int int64_t """
        self.x_shape = (10, 20)
        self.child_shape = (10, 20, 2)
        self.x_type = 'int32'
        self.info_type = 'int64'


class TestCase2(TestTDMChildOp):
    def config(self):
        """check int64_t int64_t """
        self.x_shape = (10, 20)
        self.child_shape = (10, 20, 2)
        self.x_type = 'int64'
        self.info_type = 'int64'


class TestCase3(TestTDMChildOp):
    def config(self):
        """check int64 int32 """
        self.x_shape = (10, 20)
        self.child_shape = (10, 20, 2)
        self.x_type = 'int64'
        self.info_type = 'int32'


class TestCase4(TestTDMChildOp):
    def config(self):
        """check large shape """
        self.x_shape = (100, 20)
        self.child_shape = (100, 20, 2)
        self.x_type = 'int32'
        self.info_type = 'int32'


if __name__ == "__main__":
    unittest.main()
