# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestWhereIndexOp(OpTest):
    def setUp(self):
        self.op_type = "where_index"
        self.prim_op_type = "comp"
        self.python_api = paddle.nonzero
        self.enable_cinn = False
        self.python_out_sig = ['Out']
        self.init_config()

    def test_check_output(self):
        self.check_output(check_eager=False, check_prim=True)

    def init_config(self):
        self.inputs = {
            'Condition': np.array([True, False, True]),
        }

        self.outputs = {'Out': np.array([[0], [2]], dtype='int64')}


class TestNotBool(TestWhereIndexOp):
    def init_config(self):
        self.inputs = {
            'Condition': np.array([1, 0, 8]),
        }

        self.outputs = {'Out': np.array([[0], [2]], dtype='int64')}


class TestAllFalse(TestWhereIndexOp):
    def init_config(self):
        self.inputs = {
            'Condition': np.array([False, False, False]),
        }

        self.outputs = {'Out': np.array([], dtype='int64')}


class TestRank2(TestWhereIndexOp):
    def init_config(self):
        self.inputs = {
            'Condition': np.array([[True, False], [False, True]]),
        }

        self.outputs = {'Out': np.array([[0, 0], [1, 1]], dtype='int64')}


class TestRank3(TestWhereIndexOp):
    def init_config(self):
        self.inputs = {
            'Condition': np.array(
                [
                    [[True, False], [False, True]],
                    [[False, True], [True, False]],
                    [[False, False], [False, True]],
                ]
            ),
        }

        self.outputs = {
            'Out': np.array(
                [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [2, 1, 1]],
                dtype='int64',
            )
        }


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
